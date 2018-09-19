from skills_ml.storage import ModelStorage, FSStore

import mock
import os
import tempfile
import unittest

class FakeModel(object):
    def __init__(self, val):
        self.val = val

class TestModelStorage(unittest.TestCase):
    @mock.patch('os.getcwd')
    def test_model_storage(self, mock_getcwd):
        with tempfile.TemporaryDirectory() as td:
            mock_getcwd.return_value = td
            ms = ModelStorage(FSStore(td))
            fakemodel = FakeModel(1)
            ms.save_model(fakemodel, 'test.model')
            assert set(os.listdir(os.getcwd())) == set(['test.model'])
            new_model = ms.load_model('test.model')
            assert new_model.val == fakemodel.val
