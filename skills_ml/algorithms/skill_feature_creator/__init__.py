from functools import reduce
from itertools import zip_longest
from skills_ml.algorithms.skill_feature_creator.structure_features import structFeatGeneration
from skills_ml.algorithms.skill_feature_creator.contextual_features import local_contextual_features, sent2features
import numpy as np

class FeatureCreator(object):
    """ Feature Creator Factory that help users to instantiate different
    types of feature at once and combine them together.

    Example:
        from airflow.hooks import S3Hook
        from skills_ml.algorithms.skill_feature_creator import FeatureCreator

        docs=["something", "something2"]

        feature_vector_generator = FeatureCreator(s3_conn, features="all").transform(docs)
        feature_vector_generator = FeatureCreator(s3_conn, features=["StructuralFeature", "EmbeddingFeature"]).transform(docs)

    Args:
        doc (string): job posting data.
    """
    def __init__(
            self,
            job_posting_generator,
            features="all"):
        self.all_features = [ f.__name__ for f in FeatureFactory.__subclasses__()]
        self.features = features
        self.job_posting_generator = job_posting_generator

    @property
    def selected_features(self):
        if self.features == "all":
            return self.all_features
        elif isinstance(self.features, list):
            if not (set(self.features) < set(self.all_features)):
                not_supported = list(set(self.features) - set(self.all_features))
                raise Exception("\"{}\" not supported!".format(", ".join(not_supported)))
            else:
                return self.features
        else:
            raise Exception(TypeError)

    def __iter__(self):
        feature_objects = [FeatureFactory._factory(feature) for feature in self.selected_features]

        for doc in self.job_posting_generator:
            feature_generator_map = map(lambda f: f.output(doc), feature_objects)

            # Aggregated elements from each feature generator
            agg = zip_longest(*[iter(fg) for fg in feature_generator_map])
            for f in list(map(lambda a: reduce(lambda x, y: np.concatenate((x, y), axis=1), a), agg)):
                yield f


class FeatureFactory(object):
    @staticmethod
    def _factory(type, **kwargs):
        if type == "StructuralFeature":
            return StructuralFeature()

        if type == "ContextualFeature":
            return ContextualFeature()

        if type == "EmbeddingFeature":
            return EmbeddingFeature(**kwargs)

        else:
            raise ValueError('Bad feature type \"{}\"'.format(type))


class StructuralFeature(FeatureFactory):
    """ Sturctural features in sentence level
    """
    @classmethod
    def output(self, doc):
        """ Output a feature vector.
        """
        structfeaures = structFeatGeneration(doc)
        for f in structfeaures:
            yield np.array(f).astype(np.float32)


class ContextualFeature(FeatureFactory):
    """ Contextual features in word/sentence level
    """
    @classmethod
    def output(self, doc):
        """ Output a feature vector.
        """
        contextfeaures = local_contextual_features(doc)
        # contextfeaures = sent2features(sentence)
        for f in contextfeaures:
            yield np.array(f).astype(np.float32)
        # return np.array(f).astype(np.float32)
