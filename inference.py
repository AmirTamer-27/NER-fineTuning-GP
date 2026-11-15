"""Local inference with a saved XLM-R token-classification checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from evaluation import _decode_chunk
from tokenization import tokenize_document, validate_tokenization_settings


class NERDetector:
    """Load once, then call predict(text) for each document. No gold labels needed."""

    def __init__(self, checkpoint_path, *, device='auto', batch_size=8,
                 max_length=None, stride=None):
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f'Checkpoint folder not found: {checkpoint}. Supply the downloaded best_model folder.')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        # Training saves the run configuration beside best_model, when available.
        settings = {}
        run_config = checkpoint.parent / 'config.json'
        if run_config.is_file():
            settings = json.loads(run_config.read_text(encoding='utf-8')).get('data', {})
        self.max_length = max_length if max_length is not None else settings.get('max_length', 256)
        self.stride = stride if stride is not None else settings.get('stride', 32)
        self.batch_size = batch_size
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, local_files_only=True)
        validate_tokenization_settings(self.tokenizer, self.max_length, self.stride)
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint, local_files_only=True)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        if 'O' not in self.id2label.values() or any(
            tag != 'O' and not (tag.startswith(('B-', 'I-')) and len(tag) > 2)
            for tag in self.id2label.values()
        ):
            raise ValueError('Checkpoint must contain the trained BIO label names in config.json, not generic LABEL_0 names.')
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, text):
        """Return [{start, end, label, text}]; end offsets are exclusive.

        Uses evaluation's chunk decoder and exact-span deduplication. Different
        overlapping spans are retained, just as in the current evaluator.
        """
        if not isinstance(text, str):
            raise TypeError('text must be a string.')
        if not text.strip():
            return []
        chunks = tokenize_document(
            text, self.tokenizer, max_length=self.max_length, stride=self.stride,
        )
        spans = set()
        for first in range(0, len(chunks['input_ids']), self.batch_size):
            indices = range(first, min(first + self.batch_size, len(chunks['input_ids'])))
            inputs = [{name: chunks[name][i] for name in ('input_ids', 'attention_mask')}
                      for i in indices]
            batch = self.tokenizer.pad(inputs, padding=True, return_tensors='pt')
            outputs = self.model(**{name: value.to(self.device) for name, value in batch.items()})
            rows = outputs.logits.argmax(dim=-1).cpu().tolist()
            for i, predicted in zip(indices, rows):
                # Decoder labels are only a visibility mask here; none go to the model.
                example = {'offset_mapping': chunks['offset_mapping'][i],
                           'labels': [-100 if special or start == end else 0
                                      for special, (start, end) in zip(chunks['special_tokens_mask'][i], chunks['offset_mapping'][i])]}
                spans.update(_decode_chunk(predicted, example, self.id2label, text))
        return [{'start': start, 'end': end, 'label': label, 'text': text[start:end]}
                for start, end, label in sorted(spans)]


def predict_document(text, checkpoint_path, **settings):
    """Convenience function; reuse NERDetector for multiple documents to avoid reloads."""
    return NERDetector(checkpoint_path, **settings).predict(text)
