"""Embedding model class inherited the interface from gensim"""
from skills_ml.storage import Store, FSStore

class Base2VecModel(object):
    def __init__(self, storage=None):
        self.storage = FSStore() if storage is None else storage

    def load_model(self):
        raise NotImplementedError

    def write_model(self):
        raise NotImplementedError

    def infer_vector(self):
        raise NotImplementedError
