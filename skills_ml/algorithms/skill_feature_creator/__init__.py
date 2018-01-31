class FeatureCreator(object):
    """ Feature Creator Factory that help users to instantiate different
    types of feature at once and combine them together.

    Example:
        doc = "some job posting..."
        feature_vector = FeatureCreate(doc).combine(features="all")
        feature_vector = FeatureCreate(doc).combine(features=["StructuralFeature", "EmbeddingFeature"])

    Args:
        doc (string): job posting data.
    """
    def __init__(self, doc):
        self.all_features = [ f.__name__ for f in FeatureCreator.__subclasses__()]
        self.doc = doc

    @staticmethod
    def _factory(type, doc):
        if type == "StructuralFeature":
            return StructuralFeature(doc)

        if type == "ContextualFeature":
            return ContextualFeature(doc)

        if type == "EmbeddingFeature":
            return EmbeddingFeature(doc)

    def combine(self, features="all"):
        """ Methods to combine different type of features.
        We might want to have different way to combine vectors in the future (eg. concat, average...).

        Args:
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

        result = []
        for feature in features_to_be_combined:
            result.extend(FeatureCreator._factory(feature, self.doc).output())
        return result


class StructuralFeature(FeatureCreator):
    """ Sturctural features
    """
    def output(self):
        """ Output a feature vector. Now it's just a simple example.
        """
        return ["StructuralFeature" + "_" + self.doc]


class ContextualFeature(FeatureCreator):
    """ Contextual features
    """
    def output(self):
        """ Output a feature vector. Now it's just a simple example.
        """
        return ["ContextualFeature" + "_" + self.doc]


class EmbeddingFeature(FeatureCreator):
    """ Embedding features
    """
    def output(self):
        """ Output a feature vector. Now it's just a simple example.
        """
        return ["EmbeddingFeature" + "_" + self.doc]

