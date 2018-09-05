"""Embedding model class inherited the interface from gensim"""
from skills_ml.storage import FSStore
from skills_ml.algorithms.embedding.base import ModelStorage

from gensim.models import Doc2Vec, Word2Vec
from gensim.models.fasttext import FastText as FT_gensim

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import logging

class Word2VecModel(ModelStorage, Word2Vec):
    """The Word2VecModel Object is a base object which specifies which word-embeding model.
    Inherited from gensim's Word2Vec model.

    Example:
    ```
    from skills_ml.algorithms.embedding.base import Word2VecModel

    word2vec_model = Word2VecModel()
    ```
    """
    def __init__(self, *args, **kwargs):
        ModelStorage.__init__(self, storage=kwargs.pop('storage', None))
        Word2Vec.__init__(self, *args, **kwargs)
        self.model_name = ""
        self.model_type = "word2vec"
        self._metadata = None

    def infer_vector(self, doc_words):
        """
        Average all the word-vectors together and ignore the unseen words
        Arg:
            doc_words (list): a list of tokenized words
        Returns:
            a vector representing a whole doc/sentence
        """
        sum_vector = np.zeros(self.vector_size)
        words_in_vocab = []
        for token in doc_words:
            try:
                sum_vector += self[token]
                words_in_vocab.append(token)
            except KeyError as e:
                # logging.warning("".join([str(e), ". Ignore the word."]))
                pass

        if len(words_in_vocab) == 0:
            logging.warning("None of the words is in vocabulary.")
            return np.random.rand(self.vector_size)
        sentence_vector = sum_vector / len(words_in_vocab)
        return sentence_vector

    # @property
    # def metadata(self):
    #     meta_dict = {"embedding_model": {}}
    #     meta_dict['embedding_model']['model_type'] = self.model_type
    #     meta_dict['embedding_model']['hyperparameters'] = self.__dict__
    #     meta_dict['embedding_model']['gensim_version'] = gensim_name + gensim_version
    #     return meta_dict


class Doc2VecModel(ModelStorage, Doc2Vec):
    """The Doc2VecModel Object is a base object which specifies which word-embeding model.
    Inherited from gensim's Doc2Vec model.

    Example:
    ```
    from skills_ml.algorithms.embedding.base import Doc2VecModel

    doc2vec_model = Doc2VecModel()
    ```
    """
    def __init__(self, *args, **kwargs):
        ModelStorage.__init__(self, storage=kwargs.pop('storage', None))
        Doc2Vec.__init__(self, *args, **kwargs)
        self.model_name = ""
        self.model_type = "doc2vec"
        self.lookup_dict = None

    # @property
    # def metadata(self):
    #     meta_dict = {"embedding_model": {}}
    #     meta_dict['embedding_model']['model_type'] = self.model_type
    #     meta_dict['embedding_model']['hyperparameters'] = self.__dict__
    #     meta_dict['embedding_model']['gensim_version'] = gensim_name + gensim_version
    #     return meta_dict


class FastTextModel(ModelStorage, FT_gensim):
    """The FastTextModel Object is a base object which sepcifies which word-embedding model.
    Inhereited from gensim's FastText model.

    """
    def __init__(self, *args, **kwargs):
        """

        """
        ModelStorage.__init__(self, storage=kwargs.pop('storage', None))
        FT_gensim.__init__(self, *args, **kwargs)
        self.model_name = ""
        self.model_type = "fasttext"

    def infer_vector(self, doc_words):
         """
         Average all the word-vectors together and ignore the unseen words
         """
         sum_vector = np.zeros(self.vector_size)
         words_in_vocab = []
         for token in doc_words:
             try:
                 sum_vector += self[token]
                 words_in_vocab.append(token)
             except KeyError as e:
                 # logging.warning("".join([str(e), ". Ignore the word."]))
                 pass

         if len(words_in_vocab) == 0:
             logging.warning("None of the words is in vocabulary.")
             return np.random.rand(self.vector_size)
         sentence_vector = sum_vector / len(words_in_vocab)
         return sentence_vector

    # @property
    # def metadata(self):
    #     meta_dict = {"embedding_model": {}}
    #     meta_dict['embedding_model']['model_type'] = self.model_type
    #     meta_dict['embedding_model']['hyperparameters'] = self.__dict__
    #     meta_dict['embedding_model']['gensim_version'] = gensim_name + gensim_version
    #     return meta_dict


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def fit(self, X, y):
        return self

    def transform(self, X):
        trans_X = []
        for x in X:
            trans_X.append(self.embedding_model.infer_vector(x))
        return trans_X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
