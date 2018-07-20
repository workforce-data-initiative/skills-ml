"""Embedding model class inherited the interface from gensim"""
from skills_ml.storage import FSStore
from skills_ml.algorithms.embedding.base import ModelStorage

from gensim.models import Doc2Vec, Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import logging

class Word2VecModel(ModelStorage, Word2Vec):
    """The Word2VecModel Object is a base object which specifies which word-embeding model.

    Example:
    ```
    from skills_ml.algorithms.embedding.base import Word2VecModel

    word2vec_model = Word2VecModel()
    ```
    """
    def __init__(self, *args, **kwargs):
        """
        Attributes:
            storage (:obj: `skills_ml.Store`): skills_ml Store object
            model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model.
        """
        ModelStorage.__init__(self, storage=kwargs.pop('storage', None))
        Word2Vec.__init__(self, *args, **kwargs)
        self.model_name = ""
        self._metadata = None

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
            raise KeyError("None of the words is in vocabulary.")
        sentence_vector = sum_vector / len(words_in_vocab)
        return sentence_vector


class Doc2VecModel(ModelStorage, Doc2Vec):
    """The Doc2VecModel Object is a base object which specifies which word-embeding model.

    Example:
    ```
    from skills_ml.algorithms.embedding.base import Doc2VecModel

    doc2vec_model = Doc2VecModel()
    ```
    """
    def __init__(self, *args, **kwargs):
        """
        Attributes:
            storage (:obj: `skills_ml.Store`): skills_ml Store object
            _model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model.
            lookup (dict): lookup table for mapping each jobposting index to soc code.
            training_data (np.ndarray): a document vector array where each row is a document vector.
            target (np.ndarray): a label array.
        """
        ModelStorage.__init__(self, storage=kwargs.pop('storage', None))
        Doc2Vec.__init__(self, *args, **kwargs)
        self.model_name = ""
        self.lookup_dict = None


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
