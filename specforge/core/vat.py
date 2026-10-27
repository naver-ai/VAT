# coding=utf-8
# VAT
# Copyright (c) 2026-present NAVER Corp.
# MIT License
#
# Portions derived from SpecForge (https://github.com/sgl-project/SpecForge)
# Copyright (c) 2025 sgl-project, MIT License
# OnlineVATDFlashModel.forward adapts OnlineDFlashModel.forward
# (specforge/core/dflash.py): the draft block forward, labels, weight mask,
# hard-label cross-entropy, and accuracy.

"""VAT (Verification-Aware Training) objective for DFlash drafts.

VAT replaces the DFlash block objective with three terms that all read a
simulated sequential verification of each sampled block against the target
argmax (from the target's final hidden states):

* verification-adaptive position weights: weight 1 up to and including the
  first rejected position k*, then the DFlash schedule re-anchored at k*,
  exp(-(k - k*) / loss_decay_gamma);
* a temperature-scaled KL distillation term from the frozen target head that
  shares the position weights of the CE term;
* an accept (verification) head that predicts, per position, whether the
  draft token survives sequential verification (BCE on the undecayed mask).

An optional self-coverage term (``vat_self_cov_weight > 0``, off by default)
adds, on blocks with a rejection,
``(1/R) * sum_{j >= j*} -log max(P_j, vat_self_cov_floor)``, where ``P_j`` is
the teacher-forced probability of the label prefix up to row ``j``, ``j*`` the
first rejected row, and ``R = block_size - 1``.

The accept head is registered on the draft model so the optimizer updates it
with the draft. It is training-only: ``scripts/train_dflash.py`` keeps it in
``training_state.pt`` and leaves it out of the saved draft weights.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.core.dflash import FLEX_ATTENTION_AVAILABLE, OnlineDFlashModel


class AcceptHead(nn.Module):
    """Per-position acceptance logit from the draft's output hidden state."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(hidden_states).squeeze(-1)


class OnlineVATDFlashModel(OnlineDFlashModel):
    """DFlash online training wrapper with the VAT objective."""

    def __init__(
        self,
        *,
        vat_kd_weight: float = 1.0,
        vat_kd_temperature: float = 2.0,
        vat_accept_head_weight: float = 1.0,
        vat_self_cov_weight: float = 0.0,
        vat_self_cov_floor: float = 1e-6,
        **dflash_kwargs,
    ):
        super().__init__(**dflash_kwargs)
        if not (self.loss_decay_gamma is not None and self.loss_decay_gamma > 0):
            raise ValueError(
                "VAT re-anchors the DFlash decay; set loss_decay_gamma > 0"
            )
        if min(vat_kd_weight, vat_kd_temperature, vat_accept_head_weight) <= 0:
            raise ValueError(
                "VAT KD weight/temperature and accept-head weight must be > 0"
            )
        if vat_self_cov_weight < 0 or vat_self_cov_floor <= 0:
            raise ValueError("VAT self-coverage weight must be >= 0 and its floor > 0")

        self.loss_type = "vat"
        self.vat_kd_weight = float(vat_kd_weight)
        self.vat_kd_temperature = float(vat_kd_temperature)
        self.vat_accept_head_weight = float(vat_accept_head_weight)
        self.vat_self_cov_weight = float(vat_self_cov_weight)
        self.vat_self_cov_floor = float(vat_self_cov_floor)
        # The optimizer is built over draft_model, so the head lives there.
        reference = next(self.draft_model.parameters())
        self.draft_model.accept_head = AcceptHead(
            self.draft_model.config.hidden_size
        ).to(device=reference.device, dtype=reference.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns (loss, accuracy, metrics) like ``OnlineDFlashModel.forward``.

        ``target_last_hidden_states`` is the target's final (post-norm) hidden
        state, i.e. the input of its LM head, for every input position.
        """
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )
        num_blocks = anchor_positions.shape[1]
        output_hidden = output_hidden.reshape(bsz, num_blocks, self.block_size, -1)
        logits = self.lm_head(output_hidden)

        # --- Labels: same-position prediction (position k predicts token anchor+k) ---
        label_offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_label_indices,
        )

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()
        weight_mask = weight_mask * (label_offsets > 0).float()
        weight_mask = weight_mask * torch.gather(
            loss_mask.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_label_indices,
        )

        with torch.no_grad():
            # The target state at position p predicts token p + 1.
            teacher_indices = (safe_label_indices - 1).clamp(min=0)
            teacher_hidden = torch.gather(
                target_last_hidden_states,
                1,
                teacher_indices.reshape(bsz, -1, 1).expand(
                    -1, -1, target_last_hidden_states.shape[-1]
                ),
            ).reshape(*safe_label_indices.shape, -1)
            teacher_logits = self.lm_head(teacher_hidden)
            predicted_ids = logits.argmax(dim=-1)
            # The anchor is given; the first mismatch rejects the rest of the
            # block. Slots past the sequence end never verify.
            token_match = (predicted_ids == teacher_logits.argmax(dim=-1)) & (
                valid_label_mask
            )
            token_match[..., 0] = True
            accept_len = token_match.long().cumprod(dim=-1).sum(dim=-1, keepdim=True)
            distance = (label_offsets - accept_len).clamp(min=0).float()
            loss_weights = weight_mask * torch.exp(-distance / self.loss_decay_gamma)

        # Hard CE: same-position data label.
        neg_log_q = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        loss_den = loss_weights.sum() + 1e-6
        ce_loss = (neg_log_q * loss_weights).sum() / loss_den

        # Soft target: KL(teacher || draft) at temperature T, scaled by T^2.
        temperature = self.vat_kd_temperature
        kd_per_token = F.kl_div(
            F.log_softmax(logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        kd_loss = (kd_per_token * loss_weights).sum() * temperature**2 / loss_den

        accept_logits = self.draft_model.accept_head(output_hidden)
        # Label 1 exactly on the accepted prefix (the anchor counts as accepted).
        accept_target = (label_offsets < accept_len).float()
        mask_den = weight_mask.sum()
        accept_loss = F.binary_cross_entropy_with_logits(
            accept_logits,
            accept_target,
            weight=weight_mask,
            reduction="sum",
        ) / (mask_den + 1e-6)

        loss = (
            ce_loss
            + self.vat_kd_weight * kd_loss
            + self.vat_accept_head_weight * accept_loss
        )

        if self.vat_self_cov_weight > 0:
            # Rows after the anchor; a row outside the weight mask ends the chain.
            with torch.no_grad():
                row_valid = (weight_mask[..., 1:] > 0).long().cumprod(dim=-1).bool()
                first_reject = (
                    (token_match[..., 1:] & row_valid)
                    .long()
                    .cumprod(dim=-1)
                    .sum(dim=-1, keepdim=True)
                )
                rows = torch.arange(self.block_size - 1, device=device)
                cov_gate = (rows >= first_reject) & row_valid
                cov_blocks = row_valid[..., 0].float()
            # log P_j, clipped at the floor; zero on blocks without a rejection.
            log_prefix = (-neg_log_q[..., 1:].float() * row_valid).cumsum(dim=-1)
            log_prefix = log_prefix.clamp(min=math.log(self.vat_self_cov_floor))
            self_cov_per_block = -(log_prefix * cov_gate).sum(dim=-1) / (
                self.block_size - 1
            )
            self_cov_loss = (self_cov_per_block * cov_blocks).sum() / (
                cov_blocks.sum().clamp(min=1.0)
            )
            loss = loss + self.vat_self_cov_weight * self_cov_loss

        with torch.no_grad():
            correct = (predicted_ids == target_ids) & (weight_mask > 0.5)
            accuracy = correct.sum().float() / (mask_den + 1e-6)
            block_valid = block_keep_mask.float()
            # Accepted draft tokens per block, excluding the given anchor.
            mean_accepted = (
                (accept_len.squeeze(-1) - 1).float() * block_valid
            ).sum() / (block_valid.sum() + 1e-6)

        metrics = {
            "accuracy_denom": mask_den,
            "vat/ce_loss": ce_loss.detach(),
            "vat/kd_loss": kd_loss.detach(),
            "vat/accept_loss": accept_loss.detach(),
            "vat/mean_accepted": mean_accepted,
        }
        if self.vat_self_cov_weight > 0:
            metrics["vat/self_cov_loss"] = self_cov_loss.detach()
        return loss, accuracy, metrics


__all__ = ["AcceptHead", "OnlineVATDFlashModel"]
