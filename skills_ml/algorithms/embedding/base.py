"""Embedding model class interfacing with with gensim and tensorflow"""
import os
import logging
import json
import tempfile
import numpy as np
import boto

from gensim.models import Doc2Vec, Word2Vec
import pickle

from skills_utils.s3 import download, split_s3_path, list_files
from skills_ml.storage import Store, FSStore

S3_PATH_EMBEDDING_MODEL = 'open-skills-private/model_cache/embedding/'


class Base2VecModel(object):
    def __init__(self, storage=None):
        self.storage = FSStore() if storage is None else storage

    def load_model(self):
        raise NotImplementedError

    def write_model(self):
        raise NotImplementedError

    def infer_vector(self):
        raise NotImplementedError

class Word2VecModel(Word2Vec):
    """The Word2VecModel Object is a base object which specifies which word-embeding model.

    Example:
    ```
    from skills_ml.algorithms.embedding.base import Word2VecModel

    word2vec_model = Word2VecModel()
    ```
    """
    def __init__(self, storage=None, *args, **kwargs):
        """
        Attributes:
            storage (:obj: `skills_ml.Store`): skills_ml Store object
            model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model.
        """
        super().__init__(*args, **kwargs)
        self._storage = FSStore() if storage is None else storage
        self.model_name = None
        self._metadata = None

    @property
    def storage(self):
        return self._storage

    @storage.setter
    def storage(self, value):
        if hasattr(value, 'write') and hasattr(value, 'load'):
            self._storage = value
        else:
            raise Exception(f"{value} should have methods 'write()' and 'load()'")

    @classmethod
    def load_model(cls, storage, model_name, **kwargs):
        """The method to load the model from where Storage object specified

        model_name (str): name of the model to be used.
        """
        model_loaded = storage.load(model_name)
        model = pickle.loads(model_loaded, **kwargs)
        return model

    def write_model(self, model_name=None):
        """The method to write the model to where the Storage object specified

        model_name (str): name of the model to be used.
        """
        if model_name is None:
            model_name = self.model_name

        model_pickled = pickle.dumps(self)
        self.storage.write(model_pickled, model_name)

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
                print("".join([str(e), ". Ignore the word."]))

        if len(words_in_vocab) == 0:
            raise KeyError("None of the words is in vocabulary.")
        sentence_vector = sum_vector / len(words_in_vocab)
        return sentence_vector


class Doc2VecModel(Doc2Vec):
    """The Doc2VecModel Object is a base object which specifies which word-embeding model.

    Example:
    ```
    from skills_ml.algorithms.embedding.base import Doc2VecModel

    doc2vec_model = Doc2VecModel()
    ```
    """
    def __init__(self, storage=None, lookup=True, *args, **kwargs):
        """
        Attributes:
            storage (:obj: `skills_ml.Store`): skills_ml Store object
            _model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model.
            lookup (dict): lookup table for mapping each jobposting index to soc code.
            training_data (np.ndarray): a document vector array where each row is a document vector.
            target (np.ndarray): a label array.
        """
        super().__init__(*args, **kwargs)
        self._storage = FSStore() if storage is None else storage
        self.model_name = None
        self.lookup = lookup
        self.lookup_dict = None

    @property
    def storage(self):
        return self._storage

    @storage.setter
    def storage(self, value):
        if value.__class__.__name__ in [c.__name__ for c in Store.__subclasses__()]:
            self._storage = value
        else:
            raise Exception(f"{value} is not Store Object")

    @classmethod
    def load_model(cls, storage, model_name, **kwargs):
        """The method to load the model from where Storage object specified

        model_name (str): name of the model to be used.
        """
        model_loaded = storage.load(model_name)
        model = pickle.loads(model_loaded, **kwargs)
        return model

    def write_model(self, model_name=None):
        """The method to write the model to where the Storage object specified

        model_name (str): name of the model to be used.
        """
        if model_name is None:
            model_name = self.model_name

        model_pickled = pickle.dumps(self)
        self.storage.write(model_pickled, model_name)

    def load_lookup(self, lookup_name):
        """The method to download the lookup dictionary from S3 and load to the memory.

        Returns:
            dict: a lookup table for mapping gensim index to soc code.
        """
        lookup_name = 'lookup_' + self.model_name + '.json'
        self.lookup = self.storage.load(lookup_name)
