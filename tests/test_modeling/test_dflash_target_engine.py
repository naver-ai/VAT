# coding=utf-8
# VAT
# Copyright (c) 2026-present NAVER Corp.
# MIT License

"""The DFlash target engines return the target's final hidden states for VAT.

VAT verifies drafts against ``argmax(lm_head(last_hidden_states))``, so the
returned states must be the LM head's input: the final norm output.
"""

import importlib.util
import os
import tempfile
import unittest

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

from tests.utils import get_available_port

SGLANG_MODEL = "Qwen/Qwen3-0.6B"
CAPTURE_LAYERS = [1, 9, 17, 25]
TEXT = (
    "Speculative decoding drafts several tokens with a small model and verifies "
    "them with the target model in a single forward pass. The longest prefix that "
    "matches the target's greedy choices is accepted, and decoding continues from "
    "the first rejected position. "
) * 3


def _tiny_target(path):
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(path)


class TestHFDFlashTargetEngine(unittest.TestCase):
    def test_last_hidden_states_feed_the_lm_head(self):
        from specforge.inference.target_engine.dflash_target_model import (
            get_dflash_target_model,
        )

        with tempfile.TemporaryDirectory() as path:
            _tiny_target(path)
            input_ids = torch.randint(0, 128, (2, 17))
            masks = torch.ones_like(input_ids), torch.ones_like(input_ids)
            outputs = {}
            for flag in (False, True):
                engine = get_dflash_target_model(
                    path,
                    backend="hf",
                    torch_dtype=torch.float32,
                    return_last_hidden_states=flag,
                )
                engine.set_capture_layers([0, 2])
                outputs[flag] = engine.generate_dflash_data(input_ids, *masks)
            model = engine.model

        self.assertIsNone(outputs[False].last_hidden_states)
        torch.testing.assert_close(
            outputs[True].hidden_states, outputs[False].hidden_states
        )
        self.assertEqual(outputs[True].hidden_states.shape, (2, 17, 64))
        last = outputs[True].last_hidden_states
        self.assertEqual(last.shape, (2, 17, 32))
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits
            torch.testing.assert_close(model.lm_head(last), logits)


def _sglang_worker(rank, port, result_path):
    os.environ.update(
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="1",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
    )
    from specforge.distributed import destroy_distributed, init_distributed
    from specforge.inference.target_engine.dflash_target_model import (
        get_dflash_target_model,
    )

    init_distributed(tp_size=1)
    tokenizer = AutoTokenizer.from_pretrained(SGLANG_MODEL)
    input_ids = torch.tensor(
        [tokenizer(text)["input_ids"][:96] for text in (TEXT, TEXT[40:])]
    ).cuda()
    masks = torch.ones_like(input_ids), torch.ones_like(input_ids)

    engine = get_dflash_target_model(
        SGLANG_MODEL,
        backend="sglang",
        torch_dtype=torch.bfloat16,
        return_last_hidden_states=True,
        attention_backend="flashinfer",
        mem_fraction_static=0.3,
    )
    engine.set_capture_layers(CAPTURE_LAYERS)
    output = engine.generate_dflash_data(input_ids, *masks)
    # The plain DFlash capture of the same runner, without the final states.
    _, plain = engine._backend.extend_dflash(input_ids, *masks)
    torch.save(
        {
            "input_ids": input_ids.cpu(),
            "hidden_states": output.hidden_states.cpu(),
            "last_hidden_states": output.last_hidden_states.cpu(),
            "plain_hidden_states": torch.stack(list(plain)).cpu(),
        },
        result_path,
    )
    destroy_distributed()


def _cached(model):
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model, local_files_only=True)
        return True
    except Exception:
        return False


@unittest.skipUnless(
    torch.cuda.is_available()
    and importlib.util.find_spec("sglang") is not None
    and _cached(SGLANG_MODEL),
    f"needs CUDA, sglang and a cached {SGLANG_MODEL}",
)
class TestSGLangDFlashTargetEngine(unittest.TestCase):
    def test_last_hidden_states_match_the_hf_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = os.path.join(tmp, "capture.pt")
            mp.spawn(
                _sglang_worker,
                args=(get_available_port(), result_path),
                nprocs=1,
                join=True,
            )
            captured = torch.load(result_path)

        # Returning the final states leaves the DFlash context unchanged.
        self.assertTrue(
            torch.equal(captured["hidden_states"], captured["plain_hidden_states"])
        )

        model = AutoModelForCausalLM.from_pretrained(
            SGLANG_MODEL, dtype=torch.bfloat16
        ).cuda()
        input_ids = captured["input_ids"].cuda()
        with torch.no_grad():
            reference = model(
                input_ids=input_ids, output_hidden_states=True, use_cache=False
            )
            sglang_logits = model.lm_head(captured["last_hidden_states"].cuda())

        def rel(a, b):
            return ((a.float() - b.float()).norm() / b.float().norm()).item()

        context = torch.cat(
            [reference.hidden_states[i + 1] for i in CAPTURE_LAYERS], dim=-1
        )
        # bf16 kernels differ between SGLang and HF; a pre-norm or misaligned
        # state would be off by far more.
        self.assertLess(rel(captured["hidden_states"].cuda(), context), 0.05)
        self.assertLess(
            rel(captured["last_hidden_states"].cuda(), reference.hidden_states[-1]),
            0.05,
        )
        agreement = (
            (sglang_logits.argmax(-1) == reference.logits.argmax(-1)).float().mean()
        )
        self.assertGreater(agreement.item(), 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
