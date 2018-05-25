from skills_ml.algorithms.embedding.train import EmbeddingTrainer
from skills_ml.algorithms.embedding.models import Word2VecModel, Doc2VecModel
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.job_postings.corpora.basic import Doc2VecGensimCorpusCreator, Word2VecGensimCorpusCreator

from numpy.testing import assert_array_equal, assert_almost_equal

import unittest

import logging
logging.getLogger('boto').setLevel(logging.CRITICAL)


class TestEmbeddingModels(unittest.TestCase):
    def test_word2vec(self):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        job_postings_generator = JobPostingCollectionSample(num_records=50)
        corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
        w2v = Word2VecModel(size=10, min_count=3, iter=4, window=6, workers=3)
        trainer = EmbeddingTrainer(corpus_generator, w2v)
        trainer.train()

        v1 = w2v.infer_vector(["media"])
        v2 = w2v.infer_vector(["media"])

        assert_array_equal(v1, v2)

        # test unseen vocab
        self.assertRaises(KeyError, lambda: w2v.infer_vector(["sports"]))

        # test a list that has some words not in vocab
        sentence_with_unseen_word = ["sports", "news", "and", "media"]
        sentecne_without_unseen_word = ["news", "and", "media"]
        assert_array_equal(w2v.infer_vector(sentence_with_unseen_word), w2v.infer_vector(sentecne_without_unseen_word))

    def test_doc2vec(self):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        job_postings_generator = JobPostingCollectionSample(num_records=50)
        corpus_generator = Doc2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
        d2v = Doc2VecModel(size=10, min_count=1, dm=0, alpha=0.025, min_alpha=0.025)
        trainer = EmbeddingTrainer(corpus_generator, d2v)
        trainer.train()

        # Since the inference of doc2vec is an non-deterministic algorithm, we need to reset the random seed for testing.
        d2v.random.seed(0)
        v1 = d2v.infer_vector(["media", "news"])
        d2v.random.seed(0)
        v2 = d2v.infer_vector(["media", "news"])
        assert_array_equal(v1, v2)

        # test unssen vocab
        self.assertRaises(KeyError, lambda: d2v["sports"])

        # test unseen sentence
        v1 = d2v.infer_vector(["sports"])
        v2 = d2v.infer_vector(["sports"])
        assert_array_equal(v1, v2)

