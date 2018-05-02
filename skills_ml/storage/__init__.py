from skills_utils.s3 import split_s3_path
import s3fs
from sklearn.externals import joblib
import os
import json
import io
import logging
import tempfile
from collections.abc import MutableMapping

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


class S3Store(Store):
    def __init__(self, path):
        super().__init__(path=path)

    def exists(self, fname):
        s3 = s3fs.S3FileSystem()
        return s3.exists(os.path.join(self.path, fname))

    def write(self, bytes_obj, fname):
        s3 = s3fs.S3FileSystem()
        with s3.open(os.path.join(self.path, fname), 'wb') as f:
            f.write(bytes_obj)

    def load(self, fname):
        s3 = s3fs.S3FileSystem()
        with s3.open(os.path.join(self.path, fname), 'rb') as f:
            return f.read()

    def delete(self, fname):
        s3 = s3fs.S3FileSystem()
        s3.rm(os.path.join(self.path, fname))


class FSStore(Store):
    def __init__(self, path=None):
        self.path = os.getcwd() if not path else path

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
