from skills_ml.algorithms.embedding.base import BaseEmbeddingModel
from skills_ml.algorithms.embedding.models import Doc2VecModel, EmbeddingTransformer
from skills_ml.algorithms.occupation_classifiers import SOCMajorGroup
from skills_ml.ontologies.onet import majorgroupname
from skills_ml.storage import SerializedByStorage

from sklearn.pipeline import  Pipeline

from gensim.similarities.index import AnnoyIndexer

import logging
from collections import Counter, defaultdict
import pickle
import re


first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')

def convert_camel_to_lower(name):
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub( r'\1_\2', s1).lower()


class SocClassifier(object):
    """ Interface of SOC Code Classifier for computer class to use.
    """

    def __init__(self, classifier):
        self.classifier = classifier

    def predict_soc(self, tokenized_words):
        return self.classifier.predict_soc(tokenized_words)

    @property
    def name(self):
        return "soc_" + convert_camel_to_lower(self.classifier.name)

    @property
    def description(self):
        return f"SOC code classifier using {self.classifier.description}"


class CombinedClassifier(object):
    def __init__(self, embedding, classifier, **kwargs):
        self.embedding = SerializedByStorage(embedding)
        self.classifier = SerializedByStorage(classifier)
        self.target_variable = self.classifier.target_variable

    @property
    def combined(self):
        return Pipeline([
            ('tokens_to_vector', EmbeddingTransformer(self.embedding)),
            ('classify', self.classifier)
        ])

    def predict(self, tokenized_words):
        return self.combined.predict(tokenized_words)

    def predict_soc(self, tokenized_words):
        result = self.target_variable.encoder.inverse_transform(self.combined.predict(tokenized_words)), self.combined.predict_proba(tokenized_words)
        return [(predicted_class, prob) for predicted_class, prob in zip(result[0], result[1])]

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def description(self):
        return f"combined model of {self.embedding.__class__.__name__} and {self.classifier.__class__.__name__}"


class KNNDoc2VecClassifier(BaseEmbeddingModel):
    """Nearest neightbors model to classify the jobposting data into soc code.
    If the indexer is passed, then NearestNeighbors will use approximate nearest
    neighbor approach which is much faster than the built-in knn in gensim.

    Attributes:
        embedding_model (:job: `skills_ml.algorithms.embedding.models.Doc2VecModel`): Doc2Vec embedding model
        k (int): number of nearest neighbor. If k = 1, look for the soc code from single nearest neighbor.
                 If k > 1, classify the soc code by the majority vote of nearest k neighbors.
        indexer (:obj: `gensim.similarities.index`): any kind of gensim compatible indexer
    """
    def __init__(self, embedding_model, k=1, indexer=None, model_name=None, model_storage=None, **kwargs):
        if not isinstance(embedding_model, Doc2VecModel):
            raise NotImplementedError("Only support doc2vec now.")

        if not embedding_model.lookup_dict:
            raise ValueError("`lookup_dict` is empty. Re-train the embedding model with `lookup=True` ")

        super().__init__(model_name=model_name, model_storage=model_storage)
        self.model = embedding_model
        self.model_name = "knn_cls_" + self.model.model_name
        self.indexer = indexer
        self.k = k

    def build_ann_indexer(self, num_trees=100):
        """ Annoy is an open source library to search for points in space that are close to a given query point.
        It also creates large read-only file-based data structures that are mmapped into memory so that many
        processes may share the same data. For our purpose, it is used to find similarity between words or
        documents in a vector space.

        Args:
            num_trees (int): A positive integer which effects the build time and the index size.
                             A larger value will give more accurate results, but larger indexes.
                             (https://github.com/spotify/annoy)
        Returns:
            Annoy index object
        """
        logging.info('indexing the model %s', self.model_name)
        self.model.init_sims()
        annoy_index = AnnoyIndexer(self.model, num_trees)
        self.indexer = annoy_index
        return annoy_index

    def predict_soc(self, tokenized_list):
        """The method to predict the soc code a job posting belongs to.

        Args:
            tokenized_list (list): a list of words of tokenized string

        Returns:
            tuple(str, float): The predicted soc code and cosine similarity.
        """
        inferred_vectors = EmbeddingTransformer(self.model).transform(tokenized_list)
        predicted_soc = []
        predicted_prob = []
        if self.k == 1:
            for inferred_vector in inferred_vectors:
                sims = self.model.docvecs.most_similar([inferred_vector], topn=1, indexer=self.indexer)
                resultlist = list(map(lambda l: (self.model.lookup_dict[l[0]], l[1]), [(x[0], x[1]) for x in sims]))
                predicted_soc.append(resultlist[0])

        elif self.k > 1:
            for inferred_vector in inferred_vectors:
                sims = self.model.docvecs.most_similar([inferred_vector], topn=self.k, indexer=self.indexer)
                resultlist = list(map(lambda l: (self.model.lookup_dict[l[0]], l[1]), [(x[0], x[1]) for x in sims]))
                most_common = Counter([r[0] for r in resultlist]).most_common()[0]
                resultdict = defaultdict(list)
                for u, v in resultlist:
                    resultdict[u].append(v)

                predicted_soc.append((most_common[0], sum(resultdict[most_common[0]])/most_common[1]))
        else:
            raise ValueError("k should not be smaller than 1!")

        return predicted_soc

    def save(self, model_name=None):
        """The method to write the model to where the Storage object specified
        The index wouldn't be stored, so one needs to call `build_ann_indexer` once the model is loaded

        model_name (str): name of the model to be used.
        """
        tmp_annoy_index = self.indexer
        self.indexer = None
        if model_name is None:
            model_name = self.model_name

        self.model_storage.save_model(self, model_name)
        self.indexer = tmp_annoy_index

    @property
    def name(self):
        return self.__class__.__name__ + 'K' + str(self.k)

    @property
    def description(self):
        if self.k == 1:
            return "single nearest neighbors algorithm"
        elif self.k > 1:
            return f"majority vote of {self.k} nearest neighbors"

    def __getstate__(self):
        result = self.__dict__.copy()
        result['indexer'] = None
        return result
