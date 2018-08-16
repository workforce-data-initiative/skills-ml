from skills_ml.storage import S3Store, FSStore, PersistedJSONDict
from skills_utils.s3 import upload, list_files

from sklearn.externals import joblib

from moto import mock_s3
import tempfile
import os
import unittest
import s3fs
import json
import pickle

class FakeModel(object):
    def __init__(self, val):
        self.val = val

@mock_s3
class TestS3Storage(unittest.TestCase):
    def test_s3store(self):
        import boto3
        client = boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        s3 = s3fs.S3FileSystem()

        storage = S3Store(path=f"s3://fake-open-skills/model_cache")
        assert not s3.exists(storage.path) == True

        model = FakeModel('val')
        model_pickled = pickle.dumps(model)
        storage.write(model_pickled, 'for_testing.model')

        assert storage.exists("for_testing.model")

        with storage.open("s3://fake-open-skills/model_cache/for_testing_compressed.model", "wb") as f:
            joblib.dump(model, f, compress=True)

        assert storage.exists("for_testing_compressed.model")

        model_loaded = storage.load('for_testing.model')
        model_loaded = pickle.loads(model_loaded)
        assert model_loaded.val == 'val'

        fake_lookup = {'1': 1, '2': 2, '3': 3}
        fake_lookup_bytes = json.dumps(fake_lookup).encode()
        storage.write(fake_lookup_bytes, 'for_testing.json')
        assert storage.exists("for_testing.json")

        fake_lookup_loaded = json.loads(storage.load('for_testing.json').decode())
        assert fake_lookup == fake_lookup_loaded


        storage.delete('for_testing.model')
        assert not storage.exists("for_testing.model")


class TestFSStorage(unittest.TestCase):
    def test_fsstore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FSStore(tmpdir)
            model = pickle.dumps(FakeModel('val'))
            storage.write(model, 'for_testing.model')
            assert os.path.isfile(os.path.join(tmpdir, 'for_testing.model')) == storage.exists('for_testing.model') == True

            with storage.open(os.path.join(tmpdir, "for_testing_compressed.model"), "wb") as f:
                joblib.dump(model, f, compress=True)

            assert storage.exists("for_testing_compressed.model")

            model_loaded = storage.load('for_testing.model')
            model_loaded = pickle.loads(model_loaded)
            assert model_loaded.val == 'val'

            storage.delete('for_testing.model')
            assert os.path.isfile(os.path.join(tmpdir, 'for_testing.model')) == storage.exists('for_testing.model') == False


class TestPersistedJSONDict(unittest.TestCase):
    def test_fsstore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FSStore(tmpdir)

            # 1. Ensure that a new file is correctly created and saved to
            storage_one = PersistedJSONDict(storage, 'test.json')
            storage_one['key1'] = 'value1'
            storage_one['key2'] = {'nestedkey2': 'value2'}
            storage_one.save()
            assert json.load(open(os.path.join(tmpdir, 'test.json')))\
                == {'key1': 'value1', 'key2': {'nestedkey2': 'value2'}}

            # 2. Ensure that an existing file is correctly read, updated, and saved to
            storage_two = PersistedJSONDict(storage, 'test.json')
            assert 'key1' in storage_two
            assert storage_two['key1'] == 'value1'
            storage_two['key3'] = 'value3'
            storage_two.save()
            assert json.load(open(os.path.join(tmpdir, 'test.json')))\
                == {'key1': 'value1', 'key2': {'nestedkey2': 'value2'}, 'key3': 'value3'}

            # 3. Ensure that, in the same thread, updating and svaing an old one gets new chagnes too
            storage_one['key4'] = 'value4'
            storage_one.save()
            assert json.load(open(os.path.join(tmpdir, 'test.json')))\
                == {'key1': 'value1', 'key2': {'nestedkey2': 'value2'}, 'key3': 'value3', 'key4': 'value4'}

            # 4. test autosave - this will be the fourth update of this object
            storage_one.SAVE_EVERY_N_UPDATES = 4
            storage_one['key5'] = 'value5'
            assert json.load(open(os.path.join(tmpdir, 'test.json')))\
                == {'key1': 'value1', 'key2': {'nestedkey2': 'value2'}, 'key3': 'value3', 'key4': 'value4', 'key5': 'value5'}

            # 5. test length checking
            assert len(storage_one) == 5

            # 6.  test iteration
            assert sorted(
                [(key, value) for key, value in storage_one.items()],
                key=lambda x: x[0]
            ) == [
                ('key1', 'value1'),
                ('key2', {'nestedkey2': 'value2'}),
                ('key3', 'value3'),
                ('key4', 'value4'),
                ('key5', 'value5')

            ]

    @mock_s3
    def test_s3store(self):
        import boto3
        client = boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        storage = S3Store(path=f"s3://fake-open-skills/apath")

        # 1. Ensure that a new file is correctly created and saved to
        storage_one = PersistedJSONDict(storage, 'test.json')
        storage_one['key1'] = 'value1'
        storage_one['key2'] = {'nestedkey2': 'value2'}
        storage_one.save()
        loaded = json.loads(storage.load('test.json').decode())
        assert loaded == {'key1': 'value1', 'key2': {'nestedkey2': 'value2'}}

        # 2. Ensure that an existing file is correctly read, updated, and saved to
        storage_two = PersistedJSONDict(storage, 'test.json')
        assert 'key1' in storage_two
        assert storage_two['key1'] == 'value1'
        storage_two['key3'] = 'value3'
        storage_two.save()
        loaded = json.loads(storage.load('test.json').decode())
        assert loaded == {'key1': 'value1', 'key2': {'nestedkey2': 'value2'}, 'key3': 'value3'}

        # 3. Ensure that, in the same thread, updating and svaing an old one gets new chagnes too
        storage_one['key4'] = 'value4'
        storage_one.save()
        loaded = json.loads(storage.load('test.json').decode())
        assert loaded == {'key1': 'value1', 'key2': {'nestedkey2': 'value2'}, 'key3': 'value3', 'key4': 'value4'}

        # 4. test autosave - this will be the fourth update of this object
        storage_one.SAVE_EVERY_N_UPDATES = 4
        storage_one['key5'] = 'value5'
        loaded = json.loads(storage.load('test.json').decode())
        assert loaded == {'key1': 'value1', 'key2': {'nestedkey2': 'value2'}, 'key3': 'value3', 'key4': 'value4', 'key5': 'value5'}

        # 5. test length checking
        assert len(storage_one) == 5

        # 6.  test iteration
        assert sorted(
            [(key, value) for key, value in storage_one.items()],
            key=lambda x: x[0]
        ) == [
            ('key1', 'value1'),
            ('key2', {'nestedkey2': 'value2'}),
            ('key3', 'value3'),
            ('key4', 'value4'),
            ('key5', 'value5')

        ]

