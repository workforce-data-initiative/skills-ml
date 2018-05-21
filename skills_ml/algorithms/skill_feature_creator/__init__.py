from skills_ml.algorithms.string_cleaners.nlp import NLPTransforms
from skills_ml.algorithms.skill_feature_creator.structure_features import struct_features
from skills_ml.algorithms.skill_feature_creator.contextual_features import sent2features, pre_process

from abc import ABC, abstractmethod
import numpy as np
import nltk

from functools import reduce
from itertools import zip_longest

class SequenceFeatureCreator(object):
    """ Sequence Feature Creator helps users to instantiate different
    types of feature at once and combine them together into a sentence(sequence) feature array for sequence modeling.
    It's a generator that outputs a sentence array at a time. A sentence array is composed of word vectors.

    Example:
        from skills_ml.algorithms.skill_feature_creator import FeatureCreator

        feature_vector_generator = FeatureCreator(job_posting_generator, features=)
        feature_vector_generator = FeatureCreator(job_posting_generator, features=["StructuralFeature", "EmbeddingFeature"])

    Args:
        job_posting_generator (generator): job posting generator.
        sentence_tokenizer (func): sentence tokenization function
        word_tokenizer (func): word tokenization function
        features (list): list of feature types ones want to include. If it's None or by default, it includes all the feature types.

    Yield:
        sentence_array (numpy.array): an array of word vectors represents the words and punctuations in the sentence. The dimension
                                      is (# of words)*(dimension of the concat word vector)
    """
    def __init__(
            self,
            job_posting_generator,
            sentence_tokenizer=NLPTransforms().sentence_tokenize,
            word_tokenizer=NLPTransforms().word_tokenize,
            features=None):
        self.all_features = [ f.__name__ for f in FeatureFactory.__subclasses__()]
        self.features = features
        self.job_posting_generator = job_posting_generator
        self.sentence_tokenizer = sentence_tokenizer
        self.word_tokenizer = word_tokenizer

    @property
    def selected_features(self):
        if self.features is None:
            return self.all_features
        elif isinstance(self.features, list):
            if not (set(self.features) < set(self.all_features)):
                not_supported = list(set(self.features) - set(self.all_features))
                raise TypeError("\"{}\" not supported!".format(", ".join(not_supported)))
            else:
                return self.features
        else:
            raise Exception(TypeError)

    def __iter__(self):
        feature_objects = [FeatureFactory(self.sentence_tokenizer, self.word_tokenizer)._factory(feature_type) for feature_type in self.selected_features]
        for doc in self.job_posting_generator:

            feature_generator_map = map(lambda feature: feature.output(doc), feature_objects)

            # Aggregated elements from each feature generator
            agg = zip_longest(*[iter(fg) for fg in feature_generator_map])

            # Concat all the features
            sentence_array = yield from map(lambda a: reduce(lambda x, y: np.concatenate((x, y), axis=1), a), agg)

            yield sentence_array


class FeatureFactory(object):
    def __init__(self, sentence_tokenizer=None, word_tokenizer=None):
        self.sentence_tokenizer = sentence_tokenizer
        self.word_tokenizer = word_tokenizer

    def _factory(self, feature_type, **kwargs):
        for feature_object in FeatureFactory.__subclasses__():
            if feature_object.__name__ == feature_type:
                return feature_object(self.sentence_tokenizer, self.word_tokenizer)

        raise ValueError('Bad feature type \"{}\"'.format(feature_type))


class BaseFeature(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build_feature(self):
        pass

    @abstractmethod
    def output(self):
        pass


class StructuralFeature(FeatureFactory, BaseFeature):
    """ Sturctural features
    """
    def build_feature(self, doc):
        sentences = self.sentence_tokenizer(doc)
        desc_length = len(sentences)
        features = [struct_features(sentences[i], i, desc_length, self.word_tokenizer) for i in range(desc_length)]
        return features

    def output(self, doc):
        """ Output a feature vector.
        """
        structfeaures = self.build_feature(doc)
        for f in structfeaures:
            yield np.array(f).astype(np.float32)


class ContextualFeature(FeatureFactory, BaseFeature):
    """ Contextual features
    """
    def build_feature(self, doc):
        sentences = self.sentence_tokenizer(doc)
        tagged_sentences = pre_process(sentences, self.word_tokenizer)
        features = [sent2features(s) for s in tagged_sentences]
        return features

    def output(self, doc):
        """ Output a feature vector.
        """
        contextfeaures = self.build_feature(doc)
        for f in contextfeaures:
            yield np.array(f).astype(np.float32)
