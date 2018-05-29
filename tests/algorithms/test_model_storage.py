from skills_ml.algorithms.embedding.base import ModelStorage
from skills_ml.storage import FSStore

import mock
import os
import tempfile
import unittest

class TestModelStorage(unittest.TestCase):
	@mock.patch('os.getcwd')
	def test_model_storage(self, mock_getcwd):
		with tempfile.TemporaryDirectory() as td:
			mock_getcwd.return_value = td
			ms = ModelStorage(FSStore(td))
			ms.save('test.model')
			assert set(os.listdir(os.getcwd())) == set(['test.model'])

			new_ms = ms.load(FSStore(td), 'test.model')

			self.assertEqual(ms.storage.path, new_ms.storage.path)
