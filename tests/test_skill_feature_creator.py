from skills_ml.algorithms.skill_feature_creator import SequenceFeatureCreator, EmbeddingFeature
from skills_ml.algorithms.nlp import sentence_tokenize, word_tokenize
from skills_ml.algorithms.embedding.train import EmbeddingTrainer
from skills_ml.algorithms.embedding.models import Word2VecModel, Doc2VecModel

from skills_ml.job_postings.corpora import RawCorpusCreator
from skills_ml.job_postings.corpora import Doc2VecGensimCorpusCreator, Word2VecGensimCorpusCreator
from skills_ml.job_postings.common_schema import JobPostingCollectionSample

import numpy as np
from itertools import tee

import unittest


def sentence_tokenizer_gen(doc_gen):
    for doc in doc_gen:
        sentences = sentence_tokenize(doc)
        for sentence in sentences:
            yield sentence

def word_tokenizer_gen(sent_gent):
    for sent in sent_gent:
        yield word_tokenize(sent)


class TestSkillFeatureCreator(unittest.TestCase):

    def test_skill_feature(self):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        job_postings_generator = JobPostingCollectionSample(num_records=30)
        corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields, raw=True)
        w2v = Word2VecModel(size=10, min_count=0, iter=4, window=6, workers=3)
        trainer = EmbeddingTrainer(corpus_generator, w2v)
        trainer.train()

        raw = RawCorpusCreator(JobPostingCollectionSample())
        raw1, raw2 = tee(raw)

        # default
        fc = SequenceFeatureCreator(raw1, embedding_model=w2v)
        self.assertEqual(fc.selected_features, ["StructuralFeature", "ContextualFeature", "EmbeddingFeature"])
        self.assertEqual(fc.all_features, ["StructuralFeature", "ContextualFeature", "EmbeddingFeature"])

        fc = iter(fc)
        self.assertEqual(next(fc).shape[0], np.array(next(iter(word_tokenizer_gen(sentence_tokenizer_gen(raw2))))).shape[0])
        self.assertEqual(next(fc)[0].shape[0], 29)

        # Not Supported
        fc = SequenceFeatureCreator(raw1, features=["FeatureNotSupported"])
        fc = iter(fc)
        self.assertRaises(TypeError, lambda: next(fc))

    def test_structural_feature(self):
        raw = RawCorpusCreator(JobPostingCollectionSample())
        raw1, raw2 = tee(raw)
        fc = SequenceFeatureCreator(raw1, features=["StructuralFeature"])
        fc = iter(fc)
        self.assertEqual(next(fc).shape[0], np.array(next(iter(word_tokenizer_gen(sentence_tokenizer_gen(raw2))))).shape[0])
        self.assertEqual(next(fc)[0].shape[0], 2)

    def test_contextual_feature(self):
        raw = RawCorpusCreator(JobPostingCollectionSample())
        raw1, raw2 = tee(raw)
        fc = SequenceFeatureCreator(raw1, features=["ContextualFeature"])
        fc = iter(fc)
        self.assertEqual(next(fc).shape[0], np.array(next(iter(word_tokenizer_gen(sentence_tokenizer_gen(raw2))))).shape[0])
        self.assertEqual(next(fc)[0].shape[0], 17)

    def test_embedding_feature(self):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        job_postings_generator = JobPostingCollectionSample(num_records=30)
        corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields, raw=True)
        w2v = Word2VecModel(size=10, min_count=0, iter=4, window=6, workers=3)
        trainer = EmbeddingTrainer(corpus_generator, w2v)
        trainer.train()

        job_postings = RawCorpusCreator(JobPostingCollectionSample(num_records=50))
        raw1, raw2 = tee(job_postings)

        fc = SequenceFeatureCreator(
            raw1,
            sentence_tokenizer=sentence_tokenize,
            word_tokenizer=word_tokenize,
            embedding_model=w2v,
            features=["EmbeddingFeature"]
        )
        fc = iter(fc)

        self.assertEqual(next(fc).shape[0], np.array(next(iter(word_tokenizer_gen(sentence_tokenizer_gen(raw2))))).shape[0])
        self.assertEqual(next(fc)[0].shape[0], 10)

