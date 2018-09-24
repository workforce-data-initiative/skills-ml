from gensim import __version__ as gensim_version
from gensim import __name__ as gensim_name

from skills_ml.storage import ModelStorage

class BaseEmbeddingModel(object):
    def __init__(self, model_name=None, storage=None, *args, **kwargs):
        self._model_name = model_name
        self.storage = storage

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name

    def save(self):
        if self.storage:
            model_storage = ModelStorage(self.storage)
            if self.model_name:
                model_storage.save_model(self, self.model_name)
            else:
                raise AttributeError("'self.model_name' shouldn't be {self.model_name}")
        else:
            raise AttributeError("'self.model_storage' shouldn't be {self.model_storage}")

    @classmethod
    def load(cls, model_storage, model_name, **kwargs):
        return model_storage.load_model(model_name, **kwargs)

    @property
    def metadata(self):
        meta_dict = {"embedding_model": {}}
        meta_dict['embedding_model']['model_type'] = self.model_type
        meta_dict['embedding_model']['hyperparameters'] = self.__dict__
        meta_dict['embedding_model']['gensim_version'] = gensim_name + gensim_version
        return meta_dict


