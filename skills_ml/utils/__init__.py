"""Common utilities"""

from skills_utils.s3 import split_s3_path
import s3fs
from sklearn.externals import joblib
import os
import json

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

    def exists(self):
        s3 = s3fs.S3FileSystem()
        return s3.exists(self.path)

    def write(self, obj, fname):
        s3 = s3fs.S3FileSystem()
        with s3.open(os.path.join(self.path, fname), 'wb') as f:
            joblib.dump(obj, f, compress=True)

    def load(self, fname):
        s3 = s3fs.S3FileSystem()
        with s3.open(os.path.join(self.path, fname), 'rb') as f:
            return joblib.load(f)

    def delete(self, fname):
        s3 = s3fs.S3FileSystem()
        s3.rm(os.path.join(self.path, fname))


class FSStore(Store):
    def __init__(self, path=None):
        self.path = os.getcwd() if not path else path

    def exists(self):
        return os.path.isdir(self.path)

    def write(self, obj, fname):
        os.makedirs(os.path.dirname(os.path.join(self.path, fname)), exist_ok=True)
        if isinstance(obj, dict):
             with open(os.path.join(self.path, fname), 'w') as f:
                json.dump(obj, f)
        else:
            with open(os.path.join(self.path, fname), 'w+b') as f:
                joblib.dump(obj, f, compress=True)

    def load(self, fname):
        if fname.endswith('.json'):
            with open(os.path.join(self.path, fname), 'r') as f:
                return json.load(f)
        else:
            with open(os.path.join(self.path, fname), 'rb') as f:
                return joblib.load(f)

    def delete(self, fname):
        os.remove(os.path.join(self.path, fname))
