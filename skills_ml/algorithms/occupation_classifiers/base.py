import os
import logging
import json
import tempfile
import numpy as np

from gensim.models import Doc2Vec

from skills_utils.s3 import download, split_s3_path, list_files

LOCAL_CACHE_DIRECTORY = 'tmp'

class VectorModel(object):
    """The VectorModel Object is a base object which specifies which word-embeding model to be used in
       the soc code classification.

    Example:

    from airflow.hooks import S3Hook
    from skills_ml.algorithms.occupation_classifiers.base import VectorModel

    s3_conn = S3Hook().get_conn()
    vector_model = VectorModel(s3_conn=s3_conn)

    """
    def __init__(self, model_id='va_0605', model_type='gensim_doc2vec_', saved =True,
        lookup=None, model=None, s3_conn=None, s3_path='open-skills-private/model_cache/'):
        """To initialize the SocClassifier Object, the model and lookup disctionary
        will be downloaded to the tmp/ directory and loaded to the memory.

        Attributes:
            model_id (str): model id
            model_type (str): type of the model
            model_name (str): name of the model to be used.
            saved (bool): save the model or not
            lookup_name (str): name of the lookup file
            s3_path (str): the path of the model on S3.
            s3_conn (:obj: `boto.s3.connection.S3Connection`): the boto object to connect to S3.
            files (:obj: `list` of (str)): model files need to be downloaded/loaded.
            model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model.
            lookup (dict): lookup table for mapping each jobposting index to soc code.
            training_data (np.ndarray): a document vector array where each row is a document vector.
            target (np.ndarray): a label array.
        """
        self.model_id = model_id
        self.model_type = model_type
        self.model_name = self.model_type + self.model_id
        self.saved = saved
        self.lookup_name = 'lookup_' + self.model_id + '.json'
        self.s3_path = s3_path + self.model_id
        self.s3_conn = s3_conn
        self.model = self._load_model() if model == None else model
        self.lookup = self._load_lookup() if lookup == None else lookup
        self.training_data = self.model.docvecs.doctag_syn0
        self.target = self._create_target_data()

    def _load_model(self):
        """The method to download the model from S3 and load to the memory.

        Args:
            saved (bool): wether to save the model files or just load it to the memory.

        Returns:
            gensim.models.doc2vec.Doc2Vec: The word-embedding model object.
        """
        try:
            model = Doc2Vec.load(os.path.join(LOCAL_CACHE_DIRECTORY, self.model_name))
            return model

        except:
            files  = list_files(self.s3_conn, self.s3_path)
            if not self.saved:
                with tempfile.TemporaryDirectory() as td:
                    for f in files:
                        filepath = os.path.join(td, f)
                        if not os.path.exists(filepath):
                            logging.warning('calling download from %s to %s', self.s3_path + f, filepath)
                            download(self.s3_conn, filepath, os.path.join(self.s3_path, f))
                    model = Doc2Vec.load(os.path.join(td, self.model_name))

            else:
                if not os.path.isdir(LOCAL_CACHE_DIRECTORY):
                    os.mkdir(LOCAL_CACHE_DIRECTORY)
                for f in files:
                    filepath = os.path.join(LOCAL_CACHE_DIRECTORY, f)
                    if not os.path.exists(filepath) and self.saved:
                        logging.warning('calling download from %s to %s', self.s3_path + f, filepath)
                        download(self.s3_conn, filepath, os.path.join(self.s3_path, f))
                model = Doc2Vec.load(os.path.join(LOCAL_CACHE_DIRECTORY, self.model_name))

            return model

    def _load_lookup(self):
        """The method to download the lookup dictionary from S3 and load to the memory.

        Returns:
            dict: a lookup table for mapping gensim index to soc code.
        """
        try:
            filepath = os.path.join(LOCAL_CACHE_DIRECTORY, self.lookup_name)
            with open(filepath, 'r') as handle:
                lookup = json.load(handle)
            return lookup
        except:

            if not self.saved:
                with tempfile.TemporaryDirectory() as td:
                    filepath = os.path.join(td, self.lookup_name)
                    print(filepath)
                    logging.warning('calling download from %s to %s', self.s3_path + self.lookup_name, filepath)
                    download(self.s3_conn, filepath, os.path.join(self.s3_path, self.lookup_name))
                    with open(filepath, 'r') as handle:
                        lookup = json.load(handle)

            else:
                filepath = os.path.join(LOCAL_CACHE_DIRECTORY, self.lookup_name)
                if not os.path.exists(filepath):
                    logging.warning('calling download from %s to %s', self.s3_path + self.lookup_name, filepath)
                    download(self.s3_conn, filepath , os.join(self.s3_path, self.lookup_name))
                with open(filepath, 'r') as handle:
                    lookup = json.load(handle)

            return lookup

    def _create_target_data(self):
        """To create a label array by mapping each doc vector to the lookup table.

        Returns:
            np.ndarray: label array.
        """
        y = []
        for i in range(len(self.training_data)):
            y.append(self.lookup[self.model.docvecs.index_to_doctag(i)])

        return np.array(y)
