from skills_ml.storage import open_sesame, ModelStorage, S3Store, FSStore, PersistedJSONDict, ProxyObjectWithStorage, SerializedByStorage
from skills_ml.algorithms.preprocessing import IterablePipeline
from skills_ml.algorithms.string_cleaners import nlp

from skills_utils.s3 import upload, list_files

from sklearn.externals import joblib

from functools import partial
from moto import mock_s3
import tempfile
import mock
import os
import unittest
import s3fs
import json
import numpy as np
import dill as pickle
pickle.settings['byref'] = True

class FakeModel(object):
    def __init__(self, val):
        self.val = val
        self.model_name = 'fake_model'

    def infer_vector(self, doc_words):
        return [1, 2, 3, 4]

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

        with storage.open("for_testing_compressed.model", "wb") as f:
            joblib.dump(model, f, compress=True)

        assert storage.exists("for_testing_compressed.model")


        with open_sesame("s3://fake-open-skills/model_cache/for_testing_compressed.model", "rb") as f:
            model_loaded = joblib.load(f)
        assert model.val == model_loaded.val


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
            model = FakeModel('val')
            model_pickled = pickle.dumps(model)
            storage.write(model_pickled, 'for_testing.model')
            assert os.path.isfile(os.path.join(tmpdir, 'for_testing.model')) == storage.exists('for_testing.model') == True

            with storage.open("for_testing_compressed.model", "wb") as f:
                joblib.dump(model, f, compress=True)

            assert storage.exists("for_testing_compressed.model")

            with open_sesame(os.path.join(tmpdir, "for_testing_compressed.model"), "rb") as f:
                model_loaded = joblib.load(f)
            assert model.val ==  model_loaded.val

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


class TestModelStorage(unittest.TestCase):
    @mock.patch('os.getcwd')
    def test_model_storage(self, mock_getcwd):
        with tempfile.TemporaryDirectory() as td:
            mock_getcwd.return_value = td
            ms = ModelStorage(FSStore(td))
            fake = FakeModel(1)
            ms.save_model(fake, 'test.model')
            assert set(os.listdir(os.getcwd())) == set(['test.model'])
            new_model = ms.load_model('test.model')
            assert new_model.val == fake.val


class TestSerializedByStorage(unittest.TestCase):
    @mock_s3
    def test_pickle_s3(self):
        import boto3
        client = boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        s3 = S3Store('fake-open-skills/models')
        model_storage = ModelStorage(storage=s3)
        fake = FakeModel('fake')
        model_storage.save_model(fake, fake.model_name)

        s_fake = SerializedByStorage(fake, s3, fake.model_name)
        s3.write(pickle.dumps(s_fake), 'fake.pickle')
        fake_unpickled = pickle.loads(s3.load('fake.pickle'))
        # make sure the fake model wasn't pickled but the reference
        assert fake_unpickled._model == None
        assert fake_unpickled.storage.path == s3.path
        assert fake_unpickled.val == fake.val

        # if the object to be pickled doesn't have storage attribute and didn't provide the storage
        # to SerializedByStorage, it will be serialized normally
        s_fake = SerializedByStorage(model=fake, model_name=fake.model_name)
        s3.write(pickle.dumps(s_fake), 'fake.pickle')
        fake_unpickled = pickle.loads(s3.load('fake.pickle'))
        assert fake_unpickled._model != None

    def test_delegation(self):
        fake = FakeModel('fake')
        s_fake = SerializedByStorage(model=fake, model_name=fake.model_name)
        assert fake.val == s_fake.val

    @mock_s3
    def test_with_iterable_pipelin(self):
        import boto3
        client=boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        s3 = S3Store('fake-open-skills/models')
        model_storage = ModelStorage(storage=s3)
        fake = FakeModel('fake')

        model_storage.save_model(fake, fake.model_name)
        vectorize_for_pipeline = partial(nlp.vectorize, embedding_model=SerializedByStorage(storage=s3, model_name=fake.model_name, model=fake))
        pipe = IterablePipeline(vectorize_for_pipeline)

        pipe_unpickled = pickle.loads(pickle.dumps(pipe))
        # make sure the fake model wasn't pickled but the reference
        assert pipe_unpickled.functions[-1].keywords['embedding_model']._model ==  None
        assert pipe_unpickled.functions[-1].keywords['embedding_model'].storage.path == s3.path
        # The model will be loaded when it's needed
        assert list(pipe_unpickled.build([1])) == [[1, 2, 3, 4]]


class TestProxyObject(unittest.TestCase):
    def test_delegation(self):
        fake = FakeModel('fake')
        s_fake = ProxyObjectWithStorage(model_name=fake.model_name, model_obj=fake)
        assert fake.val == s_fake.val

    @mock_s3
    def test_save_load(self):
        import boto3
        client=boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        s3 = S3Store('fake-open-skills')
        model_storage = ModelStorage(storage=s3)
        fake = FakeModel('fake')

        model_storage.save_model(fake, fake.model_name)
        proxy_fake = ProxyObjectWithStorage(model_obj=fake, storage=s3, model_name=fake.model_name)

        assert proxy_fake.storage == s3

        proxy_fake_unpickled = pickle.loads(pickle.dumps(proxy_fake))
        assert proxy_fake_unpickled.val == proxy_fake.val

        model_storage.save_model(proxy_fake, 'proxy_'+ proxy_fake.model_name)
        proxy_fake_loaded= model_storage.load_model('proxy_'+ proxy_fake.model_name)

        assert proxy_fake_loaded.val == proxy_fake.val == fake.val

    @mock_s3
    def test_with_iterable_pipeline(self):
        import boto3
        client=boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        s3 = S3Store('fake-open-skills')
        model_storage = ModelStorage(s3)

        proxy_fake = ProxyObjectWithStorage(model_obj=FakeModel('fake'), storage=s3, model_name='fake')
        model_storage.save_model(proxy_fake, proxy_fake.model_name)

        vectorize_for_pipeline = partial(nlp.vectorize, embedding_model=SerializedByStorage(model=proxy_fake, model_name=proxy_fake.model_name))
        pipe = IterablePipeline(vectorize_for_pipeline)

        s3.write(pickle.dumps(pipe), 'fake.pipe')
        pipe_unpickled = pickle.loads(s3.load('fake.pipe'))

        assert list(pipe_unpickled.build([1])) == [[1, 2, 3, 4]]

    @mock_s3
    def test_with_grid_search(self):
        import boto3
        client=boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        s3 = S3Store('fake-open-skills')
        model_storage = ModelStorage(s3)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV

        gs = GridSearchCV(RandomForestClassifier(), {})
        proxy_gs = ProxyObjectWithStorage(model_obj=gs, storage=s3, model_name='rf.grid')

        X = np.random.rand(20, 2)
        y = np.random.randint(2, size=20)

        proxy_gs.fit(X, y)
        model_storage.save_model(proxy_gs, 'rf.grid')

        loaded_proxy_gs = model_storage.load_model('rf.grid')

        assert loaded_proxy_gs.storage.path == s3.path
        assert proxy_gs.predict([[5, 6]]) == gs.predict([[5, 6]])

