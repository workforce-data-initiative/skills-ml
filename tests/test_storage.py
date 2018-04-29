from skills_ml.storage import S3Store, FSStore
from skills_utils.s3 import upload, list_files

from moto import mock_s3
import tempfile
import os
import unittest
import s3fs

class SomeClass(object):
    def __init__(self, val):
        self.val = val

@mock_s3
class TestS3Storage(unittest.TestCase):
    def test_s3store(self):
        import boto3
        client = boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')

        storage = S3Store(path=f"s3://fake-open-skills/model_cache")
        assert not storage.exists() == True

        someclass = SomeClass('val')
        storage.write(someclass, 'for_testing.model')
        s3 = s3fs.S3FileSystem()
        assert s3.exists("s3://fake-open-skills/model_cache/for_testing.model")

        newVal = storage.load('for_testing.model')
        assert newVal.val == 'val'

        fake_lookup = {'1': 1, '2': 2, '3': 3}
        storage.write(fake_lookup, 'for_testing.json')
        assert s3.exists("s3://fake-open-skills/model_cache/for_testing.json")

        assert fake_lookup == storage.load('for_testing.json')

        storage.delete('for_testing.model')
        assert not s3.exists("s3://fake-open-skills/model_cache/for_testing.model")


class TestFSStorage(unittest.TestCase):
    def test_fsstore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FSStore(tmpdir)
            assert storage.exists()

            storage.write(SomeClass('val'), 'tempfile')
            assert os.path.isfile(os.path.join(tmpdir, 'tempfile'))

            newVal = storage.load('tempfile')
            assert newVal.val == 'val'

            storage.delete('tempfile')
            assert not os.path.isfile(os.path.join(tmpdir, 'tempfilel'))
