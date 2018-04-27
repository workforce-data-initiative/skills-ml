"""Embedding model class interfacing with with gensim and tensorflow"""
import os
import logging
import json
import tempfile
import numpy as np
import boto

from gensim.models import Doc2Vec, Word2Vec

from skills_utils.s3 import download, split_s3_path, list_files
from skills_ml.storage import Store, FSStore

S3_PATH_EMBEDDING_MODEL = 'open-skills-private/model_cache/embedding/'

class Word2VecModel(object):
    """The Word2VecModel Object is a base object which specifies which word-embeding model.

    Example:
    ```
    from airflow.hooks import S3Hook
    from skills_ml.algorithms.embedding.base import Word2VecModel

    s3_conn = S3Hook().get_conn()
    word2vec_model = Word2VecModel(s3_conn=s3_conn)
    ```
    """
    def __init__(self, storage=FSStore(), model=None):
        """
        Attributes:
            storage (:)
            model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model.
        """
        self.storage = storage
        self._model = model
        self.model_name = None

    @property
    def storage(self):
        return self._storage

    @storage.setter
    def storage(self, value):
        if value.__class__.__name__ in [c.__name__ for c in Store.__subclasses__()]:
            self._storage = value
        else:
            raise Exception(f"{value} is not Store Object")

    def load_model(self, model_name):
        """The method to load the model from where Storage object specified

        model_name (str): name of the model to be used.
        """
        self._model = self.storage.load(model_name)
        self.model_name = model_name

    def write_model(self, model_name):
        """The method to write the model to where the Storage object specified

        model_name (str): name of the model to be used.
        """
        self.storage.write(self._model, model_name)


class Doc2VecModel(object):
    """The Doc2VecModel Object is a base object which specifies which word-embeding model.

    Example:
    ```
    from airflow.hooks import S3Hook
    from skills_ml.algorithms.embedding.base import Doc2VecModel

    s3_conn = S3Hook().get_conn()
    doc2vec_model = Doc2VecModel(s3_conn=s3_conn)
    ```
    """
    def __init__(self, storage=FSStore(), model=None, lookup=None):
        """
        Attributes:
            model_name (str): name of the model to be used.
            lookup_name (str): name of the lookup file
            s3_path (str): the path of the model on S3.
            s3_conn (:obj: `boto.s3.connection.S3Connection`): the boto object to connect to S3.
            model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model.
            lookup (dict): lookup table for mapping each jobposting index to soc code.
            training_data (np.ndarray): a document vector array where each row is a document vector.
            target (np.ndarray): a label array.
        """
        self.model_name = None
        self.storage = storage
        self.lookup_name = 'lookup_' + self.model_name + '.json'
        self._model = model
        self.lookup = lookup
        self.target = self._create_target_data() if hasattr(self.model, 'docvecs') else None

    @property
    def storage(self):
        return self._storage

    @storage.setter
    def storage(self, value):
        print(value)
        if value.__class__.__name__ in [c.__name__ for c in Store.__subclasses__()]:
            self._storage = value
        else:
            raise Exception(f"{value} is not Store Object")

    def load_model(self, model_name):
        """The method to download the model from S3 and load to the memory.

        Returns:
            gensim.models.doc2vec.Doc2Vec: The word-embedding model object.
        """
        self._model = self.storage.load(model_name)
        self.model_name = model_name

    def load_lookup(self, lookup_name):
        """The method to download the lookup dictionary from S3 and load to the memory.

        Returns:
            dict: a lookup table for mapping gensim index to soc code.
        """
        self.lookup = self.storage.load(lookup_name)

    @property
    def training_data(self):
        if hasattr(self._model, 'docvecs'):
            return self._model.docvecs.doctag_syn0
        else:
            return None

    @property
    def target_data(self):
        return self._create_target_data

    def _create_target_data(self):
        """To create a label array by mapping each doc vector to the lookup table.

        Returns:
            np.ndarray: label array.
        """
        y = []
        for i in range(len(self.training_data)):
            y.append(self.lookup[str(self.model.docvecs.index_to_doctag(i))])

        return np.array(y)
