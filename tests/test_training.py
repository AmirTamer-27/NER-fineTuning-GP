"""Local verification using synthetic batches; no external services."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train
from evaluation import invalid_i_transitions, token_label_counts
from collections import Counter


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = nn.Module()
        self.roberta.embeddings = nn.Embedding(8, 4)
        self.roberta.encoder = nn.Module()
        self.roberta.encoder.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.classifier = nn.Linear(4, 3)

    def forward(self, input_ids, attention_mask, labels):
        x = self.roberta.embeddings(input_ids)
        for layer in self.roberta.encoder.layer:
            x = torch.tanh(layer(x))
        logits = self.classifier(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))
        return SimpleNamespace(logits=logits, loss=loss)


def batch(ids, labels):
    return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long), "labels": torch.tensor([labels])}


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyModel()
        self.batches = [batch([1, 2], [0, -100]), batch([2, 3, 4], [1, 2, 0])]

    def test_token_diagnostics_exclude_masked_positions_and_detect_orphan_i(self):
        labels = {0: "O", 1: "B-PERSON", 2: "I-PERSON", 3: "B-BANK", 4: "I-BANK"}
        confusion = Counter({(0, 0): 2, (1, 2): 1, (2, 2): 1, (0, 3): 1})
        counts = token_label_counts(confusion, labels)
        self.assertEqual(counts["O"], {"gold": 3, "predicted": 2})
        self.assertEqual(counts["I-PERSON"], {"gold": 1, "predicted": 2})
        self.assertEqual(counts["B-BANK"], {"gold": 0, "predicted": 1})
        self.assertEqual(
            invalid_i_transitions([2, 1, 2, 2, 2, 4], [0, 1, 2, -100, 0, 0], labels),
            3,
        )

    def test_validation_weighted_loss_and_no_weight_updates(self):
        before = [p.detach().clone() for p in self.model.parameters()]
        self.model.eval()
        with torch.no_grad():
            expected = (self.model(**self.batches[0]).loss.item() + 3 * self.model(**self.batches[1]).loss.item()) / 4
        result = train.run_epoch(self.model, self.batches, torch.device("cpu"))
        self.assertAlmostEqual(result["loss"], expected, places=6)
        self.assertEqual(result["supervised_tokens"], 4)
        for old, parameter in zip(before, self.model.parameters()):
            self.assertTrue(torch.equal(old, parameter))
            self.assertIsNone(parameter.grad)

    def test_head_only_updates_only_head(self):
        for p in self.model.roberta.parameters():
            p.requires_grad = False
        encoder_before = [p.detach().clone() for p in self.model.roberta.parameters()]
        head_before = self.model.classifier.weight.detach().clone()
        optimizer = train.make_optimizer(self.model, {"encoder_learning_rate": 1e-3, "head_learning_rate": 0.1})
        train.run_epoch(self.model, self.batches, torch.device("cpu"), optimizer)
        self.assertFalse(torch.equal(head_before, self.model.classifier.weight))
        for old, parameter in zip(encoder_before, self.model.roberta.parameters()):
            self.assertTrue(torch.equal(old, parameter))

    def test_no_supervision_fails_cleanly(self):
        with self.assertRaises(ValueError):
            train.run_epoch(self.model, [batch([1], [-100])], torch.device("cpu"))

    def test_freezing_strategies(self):
        fake_transformers = ModuleType("transformers")
        factory = MagicMock()
        fake_transformers.AutoModelForTokenClassification = factory
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            spec = importlib.util.spec_from_file_location("test_xlmr", ROOT / "models" / "xlmr.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for strategy, expected in [("head_only", [False]*3), ("partial", [False, False, True]), ("full", [True]*3)]:
                factory.from_pretrained.return_value = TinyModel()
                model = module.build_model({"checkpoint": "fixture", "strategy": strategy, "train_last_n_layers": 1}, {"O": 0})
                actual = [any(p.requires_grad for p in layer.parameters()) for layer in model.roberta.encoder.layer]
                self.assertEqual(actual, expected)
                self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))

    def test_upload_logs_both_splits_and_only_selected_artifacts(self):
        tracking = MagicMock()
        tracking.start_run.return_value.__enter__.return_value.info.run_id = "test-run"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"model": {"strategy": "head_only"}, "training": {"epochs": 1}, "data": {"max_length": 512, "stride": 64}, "mlflow": {"enabled": True, "tracking_uri": "https://dagshub.com/example/repo.mlflow", "experiment_name": "fixture"}}
            row = {"epoch": 1, "train_loss": 0.4, "validation_loss": 0.5}
            for name, value in {"config.json": config, "history.json": [row], "summary.json": {"parameters": 10, "trainable_parameters": 3, "data": {}}, "best_metrics.json": row, "label2id.json": {"O": 0}, "coverage_local.json": {"private": "fixture"}}.items():
                train.write_json(root / name, value)
            with patch.dict(sys.modules, {"mlflow": tracking}), patch.dict("os.environ", {"MLFLOW_TRACKING_URI": ""}):
                train.upload_to_mlflow(root)
            tracking.log_metrics.assert_any_call({"train_loss": 0.4, "validation_loss": 0.5}, step=1)
            self.assertEqual((root / "mlflow_run_id.txt").read_text().strip(), "test-run")
            uploaded = [Path(call.args[0]).name for call in tracking.log_artifact.call_args_list]
            self.assertNotIn("coverage_local.json", uploaded)
            self.assertEqual(len(uploaded), 5)


if __name__ == "__main__":
    unittest.main()
