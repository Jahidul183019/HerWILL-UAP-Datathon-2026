"""Small CPU checks for the GPU training pipeline; no checkpoint download needed."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_submission5 as training


class Submission5Tests(unittest.TestCase):
    def test_grouped_split_keeps_normalized_duplicates_together(self):
        train = pd.DataFrame({"text": [f"Post {i}" for i in range(120)] * 2,
                              "y": [i % 3 for i in range(120)] * 2})
        a, b, fit, holdout = training.make_splits(train)
        keys = train.text.map(training.normalize)
        self.assertFalse(set(keys.iloc[fit]) & set(keys.iloc[holdout]))
        self.assertFalse(set(keys.iloc[a]) & set(keys.iloc[b]))
        self.assertEqual(set(a) | set(b), set(fit))

    def test_partial_accumulation_checkpoint_and_reload(self):
        torch.set_num_threads(2)
        backend = Tokenizer(WordLevel({"<s>":0, "<pad>":1, "</s>":2,
                                      "<unk>":3, "hello":4, "world":5}, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                            unk_token="<unk>", bos_token="<s>", eos_token="</s>")
        config = XLMRobertaConfig(vocab_size=6, hidden_size=12, num_hidden_layers=1,
                                 num_attention_heads=3, intermediate_size=24,
                                 max_position_embeddings=32, num_labels=3,
                                 hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        args = argparse.Namespace(seed=42, batch_size=2, accumulation=3,
                                  max_length=8, learning_rate=1e-3)
        train = pd.DataFrame({"text":["hello world"]*13, "y":[i%3 for i in range(13)]})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "inner"
            with patch.object(training, "new_model", side_effect=lambda _: XLMRobertaForSequenceClassification(config)):
                result = training.train_phase(train, np.arange(10), np.arange(10,13),
                                              2, directory, tokenizer, args, "cpu")
            self.assertIn(result["selected_epoch"], [1, 2])
            self.assertEqual(len(result["history"]), 2)
            checkpoint = directory / f"epoch{result['selected_epoch']}"
            model = XLMRobertaForSequenceClassification.from_pretrained(checkpoint)
            scores = training.predict(model, training.loader(train.text.iloc[10:], None, tokenizer, args), "cpu")
            np.testing.assert_allclose(scores, np.load(checkpoint / "validation_logits.npy"), atol=1e-6)
            for epoch in (1, 2):
                weights = directory / f"epoch{epoch}/model.safetensors"
                self.assertEqual(weights.exists(), epoch == result["selected_epoch"])
                self.assertTrue((directory / f"epoch{epoch}/validation_logits.npy").exists())
            retention = json.loads((directory / "checkpoint_retention.json").read_text())
            self.assertEqual(retention["selected_checkpoint"], checkpoint.name)
            with self.assertRaises(FileExistsError):
                training.train_phase(train, np.arange(10), None, 1, directory, tokenizer, args, "cpu")


if __name__ == "__main__":
    unittest.main()
