# coding=utf-8
# VAT
# Copyright (c) 2026-present NAVER Corp.
# MIT License

"""End-to-end ``scripts/train_dflash.py --loss-type vat`` on two GPUs.

A tiny random Qwen3 target (with the Qwen3 tokenizer) is trained against with
the HF and the in-process SGLang target backends, reading conversations through
the on-the-fly loader, then resumed from its checkpoint.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

from tests.utils import get_available_port

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = "Qwen/Qwen3-4B"
HIDDEN, HEADS, HEAD_DIM, TARGET_LAYERS = 128, 2, 64, 4
NUM_CONVERSATIONS = 16  # 8 steps per epoch on two ranks


def _write_target(path):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=TARGET_LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).to(torch.bfloat16).save_pretrained(path)
    tokenizer.save_pretrained(path)


def _write_draft_config(path):
    with open(ROOT / "configs" / "qwen3-4b-dflash.json") as f:
        config = json.load(f)
    config.update(
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        head_dim=HEAD_DIM,
        layer_types=["full_attention"],
        max_window_layers=1,
        num_target_layers=TARGET_LAYERS,
        block_size=4,
    )
    config["dflash_config"] = {"target_layer_ids": [1, 2]}
    with open(path, "w") as f:
        json.dump(config, f)


def _write_conversations(path):
    words = "the draft proposes a block of tokens and the target verifies".split()
    with open(path, "w") as f:
        for index in range(NUM_CONVERSATIONS):
            reply = " ".join(words[(index + k) % len(words)] for k in range(30))
            row = {
                "id": str(index),
                "conversations": [
                    {"role": "user", "content": f"Describe step {index}."},
                    {"role": "assistant", "content": reply},
                ],
            }
            f.write(json.dumps(row) + "\n")


def _train(workdir, backend, num_epochs, resume=False):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        f"--master_port={get_available_port()}",
        str(ROOT / "scripts" / "train_dflash.py"),
        "--target-model-path",
        os.path.join(workdir, "target"),
        "--target-model-backend",
        backend,
        "--draft-config-path",
        os.path.join(workdir, "draft.json"),
        "--train-data-path",
        os.path.join(workdir, "data.jsonl"),
        "--output-dir",
        os.path.join(workdir, "out"),
        "--cache-dir",
        os.path.join(workdir, "cache"),
        "--chat-template",
        "qwen",
        "--on-the-fly-loader",
        "--dataloader-num-workers",
        "2",
        "--loss-type",
        "vat",
        "--loss-decay-gamma",
        "7",
        "--block-size",
        "4",
        "--num-anchors",
        "8",
        "--max-length",
        "128",
        "--attention-backend",
        "sdpa",
        "--num-epochs",
        str(num_epochs),
        "--batch-size",
        "1",
        "--learning-rate",
        "1e-3",
        "--log-interval",
        "2",
        "--save-interval",
        "4",
        "--sglang-mem-fraction-static",
        "0.2",
    ] + (["--resume"] if resume else [])
    env = dict(os.environ, PYTHONPATH=str(ROOT), TOKENIZERS_PARALLELISM="false")
    result = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800
    )
    log = result.stdout + result.stderr
    if result.returncode != 0:
        raise AssertionError(f"train_dflash.py failed:\n{log[-6000:]}")
    return log


def _vat_logs(log):
    pattern = re.compile(
        r"Step (\d+) .*Loss: ([\d.]+), Acc: [\d.]+, vat/ce_loss: ([\d.]+), "
        r"vat/kd_loss: ([\d.]+), vat/accept_loss: ([\d.]+), vat/mean_accepted: ([\d.]+)"
    )
    return [tuple(float(x) for x in match.groups()) for match in pattern.finditer(log)]


@unittest.skipUnless(torch.cuda.device_count() >= 2, "needs two GPUs")
class TestTrainDFlashVAT(unittest.TestCase):
    def _run(self, backend):
        with tempfile.TemporaryDirectory() as workdir:
            _write_target(os.path.join(workdir, "target"))
            _write_draft_config(os.path.join(workdir, "draft.json"))
            _write_conversations(os.path.join(workdir, "data.jsonl"))

            log = _train(workdir, backend, num_epochs=1)
            steps = _vat_logs(log)
            self.assertEqual([int(s[0]) for s in steps], [2, 4, 6, 8])
            for _, loss, ce, kd, accept, _ in steps:
                # loss = CE + 1.0 * KD + 1.0 * accept BCE (rounded in the log)
                self.assertAlmostEqual(loss, ce + kd + accept, delta=2e-3)
                self.assertGreater(kd, 0.0)

            checkpoint = os.path.join(workdir, "out", "epoch_1_step_8")
            weights = load_file(os.path.join(checkpoint, "model.safetensors"))
            self.assertIn("fc.weight", weights)
            self.assertFalse(any(key.startswith("accept_head.") for key in weights))
            state = torch.load(
                os.path.join(checkpoint, "training_state.pt"), weights_only=False
            )
            self.assertEqual(set(state["accept_head"]), {"head.weight", "head.bias"})
            self.assertEqual(state["accept_head"]["head.weight"].shape, (1, HIDDEN))
            for rank in range(2):
                self.assertTrue(
                    os.path.isfile(os.path.join(checkpoint, f"optimizer_rank{rank}.pt"))
                )

            resumed = _train(workdir, backend, num_epochs=2, resume=True)
            self.assertIn("Loaded VAT accept head from checkpoint", resumed)
            self.assertEqual([int(s[0]) for s in _vat_logs(resumed)], [10, 12, 14, 16])
            self.assertTrue(
                os.path.isdir(os.path.join(workdir, "out", "epoch_2_step_16"))
            )

    def test_hf_backend(self):
        self._run("hf")

    @unittest.skipUnless(importlib.util.find_spec("sglang"), "needs sglang")
    def test_sglang_backend(self):
        self._run("sglang")


if __name__ == "__main__":
    unittest.main(verbosity=2)
