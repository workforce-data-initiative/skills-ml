import os
import logging
import json
import tempfile
import yaml
import boto

from collections import Counter, defaultdict

from gensim.models import Doc2Vec

from skills_ml.algorithms.string_cleaners import NLPTransforms

from skills_utils.s3 import download, split_s3_path, list_files


class SocClassifier(object):
    """The SocClassifier Object to classify each jobposting description to O*Net SOC code.

    Example:

    from airflow.hooks import S3Hook
    from skills_ml.algorithms.occupation_classifiers.classifiers import SocClassifier

    s3_conn = S3Hook().get_conn()
    Soc = SocClassifier(s3_conn=s3_conn)
    predicted_soc = Soc.classify(jobposting, mode='top')
    """
    def __init__(self, model_id='va_0605', model_type='gensim_doc2vec_',
        lookup=None, model=None, s3_conn=None, s3_path='open-skills-private/model_cache/'):
        """To initialize the SocClassifier Object, the model and lookup disctionary
        will be downloaded to the tmp/ directory and loaded to the memory.

        Attributes:
            model_id (str): model id
            model_type (str): type of the model
            model_name (str): name of the model to be used.
            lookup_name (str): name of the lookup file
            s3_path (str): the path of the model on S3.
            s3_conn (:obj: `boto.s3.connection.S3Connection`): the boto object to connect to S3.
            files (:obj: `list` of (str)): model files need to be downloaded/loaded.
            model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model.
            lookup (dict): lookup table for mapping each jobposting index to soc code.
        """
        self.model_id = model_id
        self.model_type = model_type
        self.model_name = self.model_type + self.model_id
        self.lookup_name = 'lookup_' + self.model_id + '.json'
        self.s3_path = s3_path + self.model_id
        self.s3_conn = s3_conn
        self.files  = list_files(self.s3_conn, self.s3_path)
        self.model = self._load_model() if model == None else model
        self.lookup = self._load_lookup() if lookup == None else lookup

    def _load_model(self, saved=False):
        """The method to download the model from S3 and load to the memory.

        Args:
            saved (bool): wether to save the model files or just load it to the memory
        """
        if not saved:
            with tempfile.TemporaryDirectory() as td:
                for f in self.files:
                    filepath = os.path.join(td, f)
                    if not os.path.exists(filepath):
                        logging.warning('calling download from %s to %s', self.s3_path + f, filepath)
                        download(self.s3_conn, filepath, os.path.join(self.s3_path, f))
                model = Doc2Vec.load(os.path.join(td, self.model_name))

        else:
            if not os.path.isdir('tmp'):
                os.mkdir('tmp')
            for f in self.files:
                filepath = 'tmp/' + f
                if not os.path.exists(filepath) and saved:
                    logging.warning('calling download from %s to %s', self.s3_path + f, filepath)
                    download(self.s3_conn, filepath, os.path.join(self.s3_path, f))
            model = Doc2Vec.load('tmp/' + self.model_name)

        return model

    def _load_lookup(self, saved=False):
        """The method to download the lookup dictionary from S3 and load to the memory.
        """
        if not saved:
            with tempfile.TemporaryDirectory() as td:
                filepath = os.path.join(td, self.lookup_name)
                print(filepath)
                logging.warning('calling download from %s to %s', self.s3_path + self.lookup_name, filepath)
                download(self.s3_conn, filepath, os.path.join(self.s3_path, self.lookup_name))
                with open(filepath, 'r') as handle:
                    lookup = json.load(handle)

        else:
            filepath = 'tmp/' + self.lookup_name
            if not os.path.exists(filepath):
                logging.warning('calling download from %s to %s', self.s3_path + self.lookup_name, filepath)
                download(self.s3_conn, filepath , os.join(self.s3_path, self.lookup_name))
                with open(filepath, 'r') as handle:
                    lookup = json.load(handle)

        return lookup

    def classify(self, jobposting, mode='top'):
        """The method to predict the soc code a job posting belongs to.

        Args:
            jobposting (str): a string of cleaned, lower-cased and pre-processed job description context.
            mode (str): a flag of which method to use for classifying.

        Returns:
            tuple(str, float): The predicted soc code and cosine similarity .
        """
        inferred_vector = self.model.infer_vector(jobposting.split())
        if mode == 'top':
            sims = self.model.docvecs.most_similar([inferred_vector], topn=1)
            resultlist = list(map(lambda l: (self.lookup[l[0]], l[1]), [(x[0], x[1]) for x in sims]))
            predicted_soc = resultlist[0]
            return predicted_soc

        if mode == 'common':
            sims = self.model.docvecs.most_similar([inferred_vector], topn=10)
            resultlist = list(map(lambda l: (self.lookup[l[0]], l[1]), [(x[0], x[1]) for x in sims]))
            most_common = Counter([r[0] for r in resultlist]).most_common()[0]
            resultdict = defaultdict(list)
            for k, v in resultlist:
                resultdict[k].append(v)resultdict[k].append(v)

            predicted_soc = (most_common[0], sum(resultdict[most_common[0]])/most_common[1])
            return predicted_soc

