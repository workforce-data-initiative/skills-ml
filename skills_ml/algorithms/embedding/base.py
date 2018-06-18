"""ModelStorage class to handle gensim model storage"""
from skills_ml.storage import FSStore
import pickle

class ModelStorage(object):
    def __init__(self, storage=None):
        self._storage = FSStore() if storage is None else storage

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
    def load(cls, storage, model_name, **kwargs):
        """The method to load the model from where Storage object specified

        model_name (str): name of the model to be used.
        """
        model_loaded = storage.load(model_name)
        model = pickle.loads(model_loaded, **kwargs)
        return model

    def save(self, model_name=None):
        """The method to write the model to where the Storage object specified

        model_name (str): name of the model to be used.
        """
        if model_name is None:
            model_name = self.model_name

        model_pickled = pickle.dumps(self)
        self.storage.write(model_pickled, model_name)
