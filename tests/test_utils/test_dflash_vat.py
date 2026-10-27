# coding=utf-8
# VAT
# Copyright (c) 2026-present NAVER Corp.
# MIT License

"""Formula-level tests for the VAT objective (``--loss-type vat``).

The draft stub and an identity LM head make every argmax known in advance, so
the simulated verification, its frontier, and each loss term are checked
against hand-derived values.
"""

import math
import types
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.core.vat import OnlineVATDFlashModel

V = 8  # vocab == hidden, so the identity head returns hidden states as logits
BS = 5
SCALE = 4.0
GAMMA = 7.0


class _OneHotDraft(nn.Module):
    """Emits fixed one-hot draft states for the single sampled block."""

    def __init__(self, predicted_ids):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=V)
        self.sliding_window = None
        self.register_buffer("states", F.one_hot(predicted_ids, V).float() * SCALE)
        self.signal = nn.Parameter(torch.zeros(V))

    def forward(self, position_ids, noise_embedding, target_hidden, attention_mask):
        del position_ids, target_hidden, attention_mask
        bsz = noise_embedding.shape[0]
        return self.states.unsqueeze(0).expand(bsz, -1, -1) + self.signal


def _model(predicted_ids, **vat_kwargs):
    lm_head = nn.Linear(V, V, bias=False)
    lm_head.weight.data.copy_(torch.eye(V))
    model = OnlineVATDFlashModel(
        draft_model=_OneHotDraft(predicted_ids),
        target_lm_head=lm_head.requires_grad_(False),
        target_embed_tokens=nn.Embedding(V, V).requires_grad_(False),
        mask_token_id=V - 1,
        block_size=BS,
        attention_backend="sdpa",
        num_anchors=1,
        loss_decay_gamma=GAMMA,
        **vat_kwargs,
    )
    # One block anchored at position 0 of a length-BS sequence.
    model._sample_anchor_positions = lambda *args, **kwargs: (
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
    )
    return model


# Labels and teacher argmax per block position (position 0 is the anchor).
LABELS = torch.tensor([0, 1, 2, 3, 4])
TEACHER = torch.tensor([0, 1, 5, 6, 6])
# The draft matches the teacher at positions 1 and 3, and the labels at 1 and 2.
PREDICTED = torch.tensor([0, 1, 2, 6, 3])


def _forward(model, teacher_ids=TEACHER):
    # The teacher state at s predicts the token at s + 1.
    teacher_states = torch.zeros(1, BS, V)
    teacher_states[0, :-1] = F.one_hot(teacher_ids[1:], V).float() * SCALE
    return model(
        input_ids=LABELS.unsqueeze(0),
        hidden_states=torch.zeros(1, BS, 3),
        loss_mask=torch.ones(1, BS),
        target_last_hidden_states=teacher_states,
    )


def _ce_terms():
    logits = F.one_hot(PREDICTED, V).float() * SCALE
    return F.cross_entropy(logits, LABELS, reduction="none")


# Teacher verification accepts the anchor and position 1, so position 2 is the
# first rejection k*: full weight through k*, then exp(-(k - k*) / gamma).
WEIGHTS = torch.tensor([0.0, 1.0, 1.0, math.exp(-1 / GAMMA), math.exp(-2 / GAMMA)])
# The undecayed mask excludes only the anchor.
MASK = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])


class TestVATObjective(unittest.TestCase):
    def test_hard_ce_keeps_the_frontier_and_decays_after_it(self):
        _, accuracy, metrics = _forward(_model(PREDICTED))
        expected = (_ce_terms() * WEIGHTS).sum() / WEIGHTS.sum()
        torch.testing.assert_close(
            metrics["vat/ce_loss"], expected, rtol=1e-5, atol=1e-6
        )
        self.assertAlmostEqual(metrics["vat/mean_accepted"].item(), 1.0, places=5)
        # Accuracy stays on hard labels: positions 1 and 2 of 4 supervised.
        self.assertAlmostEqual(accuracy.item(), 0.5, places=5)

    def test_kd_is_temperature_scaled_kl_with_the_ce_weights(self):
        temperature = 3.0
        _, _, metrics = _forward(_model(PREDICTED, vat_kd_temperature=temperature))
        draft_logits = F.one_hot(PREDICTED, V).float() * SCALE
        teacher_logits = F.one_hot(TEACHER, V).float() * SCALE
        kd = F.kl_div(
            F.log_softmax(draft_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        expected = (kd * WEIGHTS).sum() / WEIGHTS.sum() * temperature**2
        torch.testing.assert_close(
            metrics["vat/kd_loss"], expected, rtol=1e-5, atol=1e-6
        )

    def test_accept_head_targets_the_accepted_prefix(self):
        model = _model(PREDICTED)
        _, _, metrics = _forward(model)
        states = F.one_hot(PREDICTED, V).float() * SCALE
        logits = model.draft_model.accept_head(states).detach()
        # Anchor and position 1 are accepted; the first rejection is position 2.
        target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
        expected = (
            F.binary_cross_entropy_with_logits(
                logits, target, weight=MASK, reduction="sum"
            )
            / MASK.sum()
        )
        torch.testing.assert_close(
            metrics["vat/accept_loss"], expected, rtol=1e-5, atol=1e-6
        )

    def test_loss_combines_the_three_terms(self):
        model = _model(PREDICTED, vat_kd_weight=0.5, vat_accept_head_weight=2.0)
        loss, _, metrics = _forward(model)
        expected = (
            metrics["vat/ce_loss"]
            + 0.5 * metrics["vat/kd_loss"]
            + 2.0 * metrics["vat/accept_loss"]
        )
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)

    def test_self_cov_sums_clipped_prefix_log_probs_from_the_first_rejection(self):
        # Rows after the anchor; row 1 (position 2) is the first rejection.
        log_prefix = -_ce_terms()[1:].cumsum(dim=0)
        for floor in (1e-6, math.exp(-5.0)):
            with self.subTest(floor=floor):
                _, _, metrics = _forward(
                    _model(PREDICTED, vat_self_cov_weight=1.0, vat_self_cov_floor=floor)
                )
                expected = -log_prefix.clamp(min=math.log(floor))[1:].sum() / (BS - 1)
                torch.testing.assert_close(
                    metrics["vat/self_cov_loss"], expected, rtol=1e-5, atol=1e-6
                )

    def test_self_cov_is_zero_without_a_rejection(self):
        _, _, metrics = _forward(
            _model(TEACHER, vat_self_cov_weight=1.0), teacher_ids=TEACHER
        )
        self.assertEqual(metrics["vat/self_cov_loss"].item(), 0.0)

    def test_self_cov_adds_to_the_loss_and_is_off_by_default(self):
        _, _, base_metrics = _forward(_model(PREDICTED))
        self.assertNotIn("vat/self_cov_loss", base_metrics)
        loss, _, metrics = _forward(_model(PREDICTED, vat_self_cov_weight=0.6))
        expected = (
            metrics["vat/ce_loss"]
            + metrics["vat/kd_loss"]
            + metrics["vat/accept_loss"]
            + 0.6 * metrics["vat/self_cov_loss"]
        )
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)

    def test_accept_head_lives_on_the_draft_and_trains(self):
        model = _model(PREDICTED)
        self.assertIn("accept_head.head.weight", model.draft_model.state_dict())
        loss, _, _ = _forward(model)
        loss.backward()
        self.assertIsNotNone(model.draft_model.accept_head.head.weight.grad)

    def test_full_acceptance_keeps_every_position_at_full_weight(self):
        _, _, metrics = _forward(_model(TEACHER), teacher_ids=TEACHER)
        self.assertAlmostEqual(metrics["vat/mean_accepted"].item(), BS - 1, places=5)
        logits = F.one_hot(TEACHER, V).float() * SCALE
        ce = F.cross_entropy(logits, LABELS, reduction="none")
        torch.testing.assert_close(
            metrics["vat/ce_loss"], (ce * MASK).sum() / MASK.sum(), rtol=1e-5, atol=1e-6
        )

    def test_rejects_invalid_settings(self):
        cases = {
            "loss_decay_gamma > 0": {"loss_decay_gamma": None},
            "must be > 0": {"vat_kd_weight": 0.0},
            "self-coverage": {"vat_self_cov_weight": -1.0},
        }
        for message, overrides in cases.items():
            with self.subTest(message=message):
                kwargs = {"loss_decay_gamma": GAMMA, **overrides}
                with self.assertRaisesRegex(ValueError, message):
                    OnlineVATDFlashModel(
                        draft_model=_OneHotDraft(PREDICTED),
                        target_lm_head=nn.Linear(V, V, bias=False),
                        target_embed_tokens=nn.Embedding(V, V),
                        mask_token_id=V - 1,
                        block_size=BS,
                        attention_backend="sdpa",
                        **kwargs,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
