from skills_ml.algorithms.job_vectorizers.doc2vec_vectorizer import Doc2Vectorizer

class FeatureCreator(object):
    """ Feature Creator Factory that help users to instantiate different
    types of feature at once and combine them together.

    Example:
        doc = "some job posting..."
        feature_vector = FeatureCreator().combine(features="all")
        feature_vector = FeatureCreator().combine(features=["StructuralFeature", "EmbeddingFeature"])

    Args:
        doc (string): job posting data.
    """
    def __init__(
            self,
            s3_conn=None,
            embedding_model_name="gensim_doc2vec_va_0605",
            embedding_model_path="open-skills-private/model_cache/va_0605/"):
        self.all_features = [ f.__name__ for f in FeatureFactory.__subclasses__()]
        self.params = {
            "s3_conn": s3_conn,
            "embedding_model_name": embedding_model_name,
            "embedding_model_path": embedding_model_path
        }
    def combine(self, docs, features="all"):
        """ Methods to combine different type of features.
        We might want to have different way to combine vectors in the future (eg. concat, average...).

        Args:
            docs (iterator): jobposting iterator
            features (str or list): If it's "all", it will return the combined result of all types of features.
                                    If it's a list of feature types, it will return the combined result of the feature
                                    types that one specifies.
        Return:
            (list): a combined feature vector
        """
        if features == "all":
            features_to_be_combined = self.all_features
        elif isinstance(features, list):
            if not (set(features) < set(self.all_features)):
                not_supported = list(set(features) - set(self.all_features))
                raise Exception("{} not supported!".format(", ".join(not_supported)))
            else:
                features_to_be_combined = features
        else:
            raise Exception(TypeError)

        feature_objects = [FeatureFactory._factory(feature, **self.params) for feature in features_to_be_combined]
        for doc in docs:
            yield list(map(lambda x: x.output(doc), feature_objects))


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
            raise ValueError('Bad feature type {}'.format(type))

class StructuralFeature(FeatureFactory):
    """ Sturctural features
    """
    def output(self, doc):
        """ Output a feature vector. Now it's just a simple example.
        """
        return ["StructuralFeature_" + doc]


class ContextualFeature(FeatureFactory):
    """ Contextual features
    """
    def output(self, doc):
        """ Output a feature vector. Now it's just a simple example.
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
        return self.embedding_model.model.infer_vector(doc)
