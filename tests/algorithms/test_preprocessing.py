from skills_ml.algorithms.preprocessing import IterablePipeline
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.algorithms.string_cleaners.nlp import fields_join, clean_html, clean_str, sentence_tokenize, word_tokenize

from functools import partial, update_wrapper
import unittest

class TestIterablePipeline(unittest.TestCase):
    def setUp(self):
        self.jp = list(JobPostingCollectionSample())

    def test_iterable_pipeline(self):
        def sentence_counter(doc):
            """count sentence for a document"""
            return  len(doc)

        updated_fields_join = partial(fields_join, document_schema_fields=['description'])
        update_wrapper(updated_fields_join, fields_join)

        pipe1 = IterablePipeline(
                updated_fields_join,
                clean_html,
                sentence_tokenize,
                clean_str,
                word_tokenize,
                sentence_counter
                )
        pipe2 = IterablePipeline(
                updated_fields_join,
                clean_html,
                sentence_tokenize,
                sentence_counter
                )

        pipe1_generator = pipe1.build(self.jp)
        pipe2_generator = pipe2.build(self.jp)

        assert list(pipe1_generator) == list(pipe2_generator)
        assert pipe1.description == [f.__doc__ for f in pipe1.functions]
