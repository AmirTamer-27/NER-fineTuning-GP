"""Checkpoint backup behavior without a remote MLflow service."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpointing import upload_best_checkpoint


class CheckpointBackupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        model = self.root / 'best_model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        (model / 'model.safetensors').write_bytes(b'fixture')
        (self.root / 'config.json').write_text('{}')
        (self.root / 'best_metrics.json').write_text('{"epoch":1}')
        self.tracking = MagicMock()

    def test_completed_backup_contains_model_and_publishes_pointer(self):
        receipt = upload_best_checkpoint(self.root, self.tracking, {'epoch': 1})
        self.tracking.log_artifacts.assert_called_once_with(
            str(self.root / 'best_model'), artifact_path='checkpoints/epoch_0001/best_model')
        self.tracking.log_dict.assert_called_once_with(receipt, 'results/latest_checkpoint.json')
        self.assertEqual(json.loads((self.root / 'checkpoint_upload.json').read_text()), receipt)
        self.assertEqual(len(self.tracking.log_artifact.call_args_list), 2)

    def test_final_upload_does_not_upload_same_weights_again(self):
        upload_best_checkpoint(self.root, self.tracking, {'epoch': 1})
        upload_best_checkpoint(self.root, self.tracking, {'epoch': 1})
        self.assertEqual(self.tracking.log_artifacts.call_count, 1)

    def test_interrupted_new_upload_preserves_previous_completed_pointer(self):
        previous = upload_best_checkpoint(self.root, self.tracking, {'epoch': 1})
        self.tracking.reset_mock()
        self.tracking.log_artifacts.side_effect = RuntimeError('connection interrupted')
        with self.assertRaisesRegex(RuntimeError, 'connection interrupted'):
            upload_best_checkpoint(self.root, self.tracking, {'epoch': 2})
        self.assertEqual(json.loads((self.root / 'checkpoint_upload.json').read_text()), previous)
        self.tracking.log_dict.assert_not_called()
        self.tracking.log_artifacts.assert_called_once_with(
            str(self.root / 'best_model'), artifact_path='checkpoints/epoch_0002/best_model')

    def test_missing_weights_are_not_advertised_as_checkpoint(self):
        (self.root / 'best_model' / 'model.safetensors').unlink()
        with self.assertRaises(FileNotFoundError):
            upload_best_checkpoint(self.root, self.tracking, {'epoch': 1})
        self.tracking.log_dict.assert_not_called()


if __name__ == '__main__':
    unittest.main()
