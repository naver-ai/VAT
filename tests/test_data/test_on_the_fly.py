# coding=utf-8
# VAT
# Copyright (c) 2026-present NAVER Corp.
# MIT License

"""``--on-the-fly-loader`` yields the samples of the upfront preprocessing."""

import socket
import unittest

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import Dataset
from specforge.data.on_the_fly import OnTheFlyDataset
from specforge.data.preprocessing import build_eagle3_dataset
from specforge.data.utils import DataCollatorWithPadding

MODEL = "Qwen/Qwen3-4B"
TEMPLATE = "qwen"
MAX_LENGTH = 96
MIN_LOSS_TOKENS = 12


def _conversation(index, reply_words):
    return [
        {"role": "user", "content": f"Question {index}: what comes next?"},
        {"role": "assistant", "content": " ".join(["token"] * reply_words)},
    ]


def _rows():
    # Rows 1 and 4 are ineligible: an empty conversation and a reply shorter
    # than MIN_LOSS_TOKENS. Row 5 is truncated at MAX_LENGTH.
    replies = [12, None, 20, 9, 2, 200, 15]
    return {
        "id": [str(index) for index in range(len(replies))],
        "conversations": [
            [] if words is None else _conversation(index, words)
            for index, words in enumerate(replies)
        ],
    }


class TestOnTheFlyDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        cls.raw = Dataset.from_dict(_rows())

    def _on_the_fly(self, dataset=None):
        return OnTheFlyDataset(
            self.raw if dataset is None else dataset,
            self.tokenizer,
            TEMPLATE,
            MAX_LENGTH,
            min_loss_tokens=MIN_LOSS_TOKENS,
        )

    def _upfront(self, shuffle_seed=42):
        upfront = build_eagle3_dataset(
            self.raw,
            self.tokenizer,
            chat_template=TEMPLATE,
            max_length=MAX_LENGTH,
            shuffle_seed=shuffle_seed,
            num_proc=1,
        )
        return upfront.filter(lambda x: x["loss_mask"].sum() >= MIN_LOSS_TOKENS)

    def test_eligible_rows_match_the_upfront_preprocessing(self):
        upfront = self._upfront()
        # build_eagle3_dataset shuffles before tokenizing; read the same order.
        on_the_fly = self._on_the_fly(self.raw.shuffle(seed=42))
        expected = [upfront[index] for index in range(len(upfront))]
        samples = [on_the_fly._sample(index) for index in range(len(on_the_fly))]
        eligible = [sample for sample in samples if sample is not None]
        self.assertEqual(len(eligible), len(self.raw) - 2)
        self.assertEqual(len(eligible), len(expected))
        for index, sample in enumerate(samples):
            if sample is not None:
                self.assertTrue(
                    torch.equal(on_the_fly[index]["input_ids"], sample["input_ids"])
                )
        for got, want in zip(eligible, expected):
            self.assertEqual(set(got), {"input_ids", "loss_mask", "attention_mask"})
            for key in got:
                self.assertEqual(got[key].shape, want[key].shape, key)
                self.assertTrue(torch.equal(got[key], want[key].to(got[key].dtype)))
        self.assertTrue(max(s["input_ids"].shape[-1] for s in eligible) <= MAX_LENGTH)

    def test_ineligible_rows_are_replaced_deterministically(self):
        dataset = self._on_the_fly()
        self.assertEqual(len(dataset), len(self.raw))
        for index in (1, 4):
            sample = dataset[index]
            self.assertGreaterEqual(int(sample["loss_mask"].sum()), MIN_LOSS_TOKENS)
            again = dataset[index]
            self.assertTrue(torch.equal(sample["input_ids"], again["input_ids"]))

    def test_no_eligible_row_fails_loudly(self):
        dataset = self._on_the_fly(self.raw.select([1, 4]))
        with self.assertRaisesRegex(ValueError, "no eligible sample"):
            dataset[0]

    def test_batches_through_dataloader_workers(self):
        # The collator reads the (single-rank) sequence-parallel group size.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0
        )
        self.addCleanup(dist.destroy_process_group)
        loader = DataLoader(
            self._on_the_fly(),
            batch_size=2,
            num_workers=2,
            collate_fn=DataCollatorWithPadding(),
        )
        batches = list(loader)
        self.assertEqual(sum(b["input_ids"].shape[0] for b in batches), len(self.raw))
        for batch in batches:
            self.assertEqual(batch["input_ids"].shape, batch["loss_mask"].shape)
            self.assertEqual(batch["input_ids"].shape, batch["attention_mask"].shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
