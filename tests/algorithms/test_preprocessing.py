from skills_ml.algorithms.preprocessing import ProcessingPipeline, IterablePipeline
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.job_postings.filtering import JobPostingFilterer
from skills_ml.algorithms.string_cleaners.nlp import vectorize, normalize, fields_join, clean_html, clean_str, sentence_tokenize, word_tokenize

from functools import partial, update_wrapper

import numpy as np
import unittest


class FakeEmbeddingModel(object):
    def __init__(self, size):
        self.size = size
        self.random_vector = np.random.rand(self.size)

    def infer_vector(self, tokenized_word):
        return self.random_vector

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.jp = list(JobPostingCollectionSample())

    @property
    def updated_fields_join(self):
        return partial(fields_join, document_schema_fields=['description'])

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

    def test_processing_pipeline(self):
        preprocessor = ProcessingPipeline(
                normalize,
                clean_html,
                clean_str,
                word_tokenize,
                )
        assert preprocessor("Why so SERIOUS?") == ["why", "so", "serious"]

    def test_combined_processing_iterable(self):
        w2v = FakeEmbeddingModel(size=10)
        vectorization = ProcessingPipeline(
                normalize,
                clean_html,
                clean_str,
                word_tokenize,
                partial(vectorize, embedding_model=w2v)
                )

        pipe_combined = IterablePipeline(
                self.updated_fields_join,
                vectorization
                )

        pipe_iterable = IterablePipeline(
                self.updated_fields_join,
                normalize,
                clean_html,
                clean_str,
                word_tokenize,
                partial(vectorize, embedding_model=w2v)
                )

        pipe_combined_generator = pipe_combined.build(self.jp)
        pipe_iterable_generator = pipe_iterable.build(self.jp)

        combined = list(pipe_combined_generator)
        iterable = list(pipe_iterable_generator)

        assert len(combined) == len(iterable)

        for c, i in zip(combined, iterable):
            np.testing.assert_array_equal(c, i)


