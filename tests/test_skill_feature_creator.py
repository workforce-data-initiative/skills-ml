from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.job_postings.corpora.basic import RawCorpusCreator
from skills_ml.algorithms.skill_feature_creator import FeatureCreator
from skills_ml.algorithms.string_cleaners import NLPTransforms

import numpy as np
from itertools import tee

import unittest

def sentence_tokenizer_gen(doc_gen):
    for doc in doc_gen:
        sentences = NLPTransforms().sentence_tokenize(doc)
        for sentence in sentences:
            yield sentence

def word_tokenizer_gen(sent_gent):
    for sent in sent_gent:
        yield NLPTransforms().word_tokenize(sent)


class TestSkillFeatureCreator(unittest.TestCase):
    def test_skill_feature(self):
        raw = RawCorpusCreator(JobPostingCollectionSample())
        raw1, raw2 = tee(raw)

        # default
        fc = FeatureCreator(raw1)
        self.assertEqual(fc.selected_features, ["StructuralFeature", "ContextualFeature"])
        self.assertEqual(fc.all_features, ["StructuralFeature", "ContextualFeature"])

        fc = iter(fc)
        self.assertEqual(next(fc).shape[0], np.array(next(iter(word_tokenizer_gen(sentence_tokenizer_gen(raw2))))).shape[0])

        #
        fc = FeatureCreator(raw1, features=["StructuralFeature"])
        fc = iter(fc)
        self.assertEqual(next(fc).shape[0], np.array(next(iter(word_tokenizer_gen(sentence_tokenizer_gen(raw2))))).shape[0])

        fc = FeatureCreator(raw1, features=["FeatureNotSupported"])
        fc = iter(fc)
        self.assertRaises(TypeError, lambda: next(fc))
