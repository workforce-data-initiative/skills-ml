from functools import reduce

from skills_ml.algorithms.job_vectorizers.doc2vec_vectorizer import Doc2Vectorizer

class FeatureCreator(object):
    """ Feature Creator Factory that help users to instantiate different
    types of feature at once and combine them together.

    Example:
        from airflow.hooks import S3Hook
        from skills_ml.algorithms.skill_feature_creator import FeatureCreator

        s3_conn = S3Hook().get_conn()

        docs=["something", "something2"]

        feature_vector_generator = FeatureCreator(s3_conn, features="all").transform(docs)
        feature_vector_generator = FeatureCreator(s3_conn, features=["StructuralFeature", "EmbeddingFeature"]).transform(docs)

    Args:
        doc (string): job posting data.
    """
    def __init__(
            self,
            s3_conn=None,
            features="all",
            embedding_model_name=None,
            embedding_model_path=None):
        self.all_features = [ f.__name__ for f in FeatureFactory.__subclasses__()]
        self.params = {
            "s3_conn": s3_conn,
            "embedding_model_name": embedding_model_name,
            "embedding_model_path": embedding_model_path
        }
        self.features = features

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

    def transform(self, docs, combine="concat"):
        """ Methods to combine different types of feature and transform to a vector as a generator.
        We might want to have different way to combine vectors in the future (eg. concat, average...).

        Args:
            docs (iterator): jobposting iterator
            features (str or list): If it's "all", it will return the combined result of all types of features.
                                    If it's a list of feature types, it will return the combined result of the feature
                                    types that one specifies.
        Return:
            (generator): a combined feature vector generator
        """
        features_to_be_combined = self.selected_features
        feature_objects = [FeatureFactory._factory(feature, **self.params) for feature in features_to_be_combined]
        if combine == "concat":
            for doc in docs:
                yield reduce(lambda x, y: x+y, map(lambda x: x.output(doc), feature_objects))
        else:
            raise Exception("\"{}\" not supported!".format(combine))


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
    """ Sturctural features
    """
    def output(self, doc):
        """ Output a feature vector. Need to be implemented! Now it's just a simple example.
        """
        return ["StructuralFeature_" + doc]


class ContextualFeature(FeatureFactory):
    """ Contextual features
    """
    def output(self, doc):
        """ Output a feature vector. Need to be implemented! Now it's just a simple example.
        """
        return ["ContextualFeature_" + doc]


class EmbeddingFeature(FeatureFactory):
    """ Embedding features
    """
    def __init__(self, **kwargs):
        super().__init__()
        model_name = kwargs["embedding_model_name"]
        path = kwargs["embedding_model_path"]
        s3_conn = kwargs["s3_conn"]
        self.embedding_model = Doc2Vectorizer(model_name, path, s3_conn)

    def output(self, doc):
        """ Output a feature vector.
        """
        return list(list(self.embedding_model.vectorize([doc]))[0])
