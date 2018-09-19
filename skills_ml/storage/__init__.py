import s3fs
import os
from os.path import dirname
import json
import logging
from collections.abc import MutableMapping
from contextlib import contextmanager
from retrying import retry
from urllib.parse import urlparse

from sklearn.externals import joblib
import dill as pickle


@retry(stop_max_delay=150000, wait_fixed=3000)
@contextmanager
def open_sesame(path, *args, **kwargs):
    path_parsed = urlparse(path)
    scheme = path_parsed.scheme

    if not scheme or scheme == 'file':
        os.makedirs(dirname(path), exist_ok=True)
        with open(path, *args, **kwargs) as f:
            yield f
    elif scheme == 's3':
        s3 = s3fs.S3FileSystem()
        with s3.open(path, *args, **kwargs) as f:
            yield f


def retry_if_io_error(exception):
    return isinstance(exception, IOError)


class Store(object):
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return f"{self.__class__.__name__}(path={self.path})"

    def __repr__(self):
        return str(self)

    def exists(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def write(self, obj):
        raise NotImplementedError

    def list(self, subpath):
        raise NotImplementedError


class S3Store(Store):
    def __init__(self, path):
        super().__init__(path=path)

    @contextmanager
    def open(self, fname, *args, **kwargs):
        s3 = s3fs.S3FileSystem()
        with s3.open(os.path.join(self.path, fname), *args, **kwargs) as f:
            yield f

    @retry(retry_on_exception=retry_if_io_error)
    def exists(self, fname):
        s3 = s3fs.S3FileSystem()
        return s3.exists(os.path.join(self.path, fname))

    @retry(retry_on_exception=retry_if_io_error)
    def write(self, bytes_obj, fname):
        s3 = s3fs.S3FileSystem()
        with s3.open(os.path.join(self.path, fname), 'wb') as f:
            f.write(bytes_obj)

    @retry(retry_on_exception=retry_if_io_error)
    def load(self, fname):
        s3 = s3fs.S3FileSystem()
        with s3.open(os.path.join(self.path, fname), 'rb') as f:
            return f.read()

    @retry(retry_on_exception=retry_if_io_error)
    def delete(self, fname):
        s3 = s3fs.S3FileSystem()
        s3.rm(os.path.join(self.path, fname))

    @retry(retry_on_exception=retry_if_io_error)
    def list(self, subpath):
        s3 = s3fs.S3FileSystem()
        return [
            k.split('/')[-1] for k in
            s3.ls(os.path.join(self.path, subpath))
        ]


class FSStore(Store):
    def __init__(self, path=None):
        self.path = os.getcwd() if not path else path

    @contextmanager
    def open(self, fname, *args, **kwargs):
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, fname), *args, **kwargs) as f:
            yield f

    def exists(self, fname):
        return os.path.isfile(os.path.join(self.path, fname))

    def write(self, bytes_obj, fname):
        os.makedirs(os.path.dirname(os.path.join(self.path, fname)), exist_ok=True)
        with open(os.path.join(self.path, fname), 'wb') as f:
            f.write(bytes_obj)

    def load(self, fname):
        with open(os.path.join(self.path, fname), 'rb') as f:
            return f.read()

    def delete(self, fname):
        os.remove(os.path.join(self.path, fname))

    def list(self, subpath):
        return os.listdir(os.path.join(self.path, subpath))


class InMemoryStore(Store):
    def __init__(self, *args, **kwargs):
        if 'path' not in kwargs and len(args) == 0:
            super().__init__(path='nothing', *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self.store = {}

    def exists(self, fname):
        return fname in self.store

    def write(self, bytes_obj, fname):
        self.store[fname] = bytes_obj

    def load(self, fname):
        return self.store[fname]

    def delete(self, fname):
        del self.store[fname]

    def list(self, subpath):
        return [key for key in self.store if key.startswith(subpath)]


class PersistedJSONDict(MutableMapping):

    SAVE_EVERY_N_UPDATES = 1000

    def __init__(self, storage, fname):
        self.fname = fname
        self.fs = storage
        if self.fs.exists(self.fname):
            loaded = self.fs.load(self.fname).decode() or '{}'
            self._storage = json.loads(loaded)
        else:
            self._storage = dict()

        self.num_updates = 0
        logging.info(f'Loaded storage with {len(self)} keys')

    def __getitem__(self, key):
        return self._storage[key]

    def __iter__(self):
        return iter(self._storage)

    def __len__(self):
        return len(self._storage)

    def __delitem__(self, key):
        del self._storage[key]

    def __setitem__(self, key, value):
        self._storage[key] = value
        self.num_updates += 1
        if self.num_updates % self.SAVE_EVERY_N_UPDATES == 0:
            logging.info(f'Auto-saving after {self.num_updates} updates')
            self.save()

    def __keytransform__(self, key):
        return key

    def __contains__(self, key):
        return key in self._storage

    def save(self):
        logging.info(f'Attempting to save storage of length {len(self)} to {self.fs.path}')
        if self.fs.exists(self.fname):
            loaded = self.fs.load(self.fname).decode() or '{}'
            saved_data = json.loads(loaded)
            logging.info(
                f'Merging {len(self)} in-memory keys with {len(saved_data)} stored keys. In-memory data takes priority'
            )
            saved_data.update(self._storage)
            self._storage = saved_data

        json_bytes = json.dumps(self._storage).encode()
        self.fs.write(json_bytes, self.fname)


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

    def load_model(self, model_name, **kwargs):
        """The method to load the model from where Storage object specified

        model_name (str): name of the model to be used.
        """
        with self.storage.open(model_name, "rb") as f:
            model = joblib.load(f)

        return model

    def save_model(self, model, model_name):
        """The method to write the model to where the Storage object specified

        model_name (str): name of the model to be used.
        """
        with self.storage.open(model_name, "wb") as f:
            joblib.dump(model, f, compress=True)


class SerializableModel(object):
    def __init__(self, model=None, storage=None, model_name=None):
        self._model = model
        self.storage = storage
        self.model_name = model_name

    def __getitem__(self, item):
        if self._model is None:
            logging.info("Model wasn't loaded yet!")
        result = self.model[item]
        return result

    def __getattr__(self, item):
        if item not in self.__dict__.keys():
            if self._model is None:
                logging.info("Model wasn't loaded yet!")
            result = getattr(self.model, item)
            return result

    @property
    def model(self):
        if self._model is None:
            logging.info(f"Loading Model-{self.model_name} from {self.storage.path}")
            self._model = ModelStorage.load_model(self.storage, self.model_name)
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def __getstate__(self):
        result = self.__dict__.copy()
        result['_model'] = None
        return result

    def __setstate__(self, state):
        self.__dict__ = state
        model_storage = state['storage']
        model_name = state['model_name']

