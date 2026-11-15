"""Offline regression checks for shared training/inference tokenization."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from tokenization import tokenize_document
from prepare_data import prepare_ner_dataset


class SharedTokenizationTests(unittest.TestCase):
    def setUp(self):
        tokenizer = Tokenizer(WordLevel(
            {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
             'Ahmed': 4, 'Ali': 5, 'أحمد': 6, 'علي': 7}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token='[PAD]', unk_token='[UNK]',
            cls_token='[CLS]', sep_token='[SEP]', model_max_length=32)

    def test_overflow_matches_previous_tokenizer_output(self):
        for text in ('Ahmed Ali ' * 10, 'أحمد علي ' * 10):
            with self.subTest(text=text[:10]):
                previous = self.tokenizer(
                    text, max_length=8, truncation=True, stride=2,
                    return_overflowing_tokens=True, return_offsets_mapping=True,
                    return_special_tokens_mask=True, padding=False)
                shared = tokenize_document(text, self.tokenizer, max_length=8, stride=2)
                self.assertEqual(dict(shared), dict(previous))
                self.assertGreater(len(shared['input_ids']), 1)
                self.assertEqual(shared['offset_mapping'][0][-3:-1],
                                 shared['offset_mapping'][1][1:3])

    def test_training_uses_shared_chunks_and_retains_bio_alignment(self):
        text = 'Ahmed Ali ' * 10
        record = {'id': 'example', 'text': text,
                  'spans': [{'start': 0, 'end': 9, 'label': 'PERSON_NAME'}]}
        shared = tokenize_document(text, self.tokenizer, max_length=8, stride=2)
        with patch('prepare_data.tokenize_document', wraps=tokenize_document) as tokenize:
            examples, labels, _ = prepare_ner_dataset(
                [record], self.tokenizer, ['PERSON_NAME'], max_length=8,
                stride=2, include_metadata=True)
        tokenize.assert_called_once()
        for i, example in enumerate(examples):
            self.assertEqual(example['input_ids'], shared['input_ids'][i])
            self.assertEqual(example['attention_mask'], shared['attention_mask'][i])
            self.assertEqual(example['offset_mapping'], [list(o) for o in shared['offset_mapping'][i]])
        self.assertEqual(examples[0]['labels'][:3],
                         [-100, labels['B-PERSON_NAME'], labels['I-PERSON_NAME']])

    def test_invalid_stride_rejected_before_tokenizing(self):
        with self.assertRaises(ValueError):
            tokenize_document('Ahmed', self.tokenizer, max_length=8, stride=6)


if __name__ == '__main__':
    unittest.main()
