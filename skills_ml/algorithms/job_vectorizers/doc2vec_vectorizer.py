import os
import logging
from gensim.models import Doc2Vec

from skills_ml.algorithms.string_cleaners import NLPTransforms

from skills_utils.s3 import download

MODEL_NAME = 'gensim_doc2vec_va_0605'
PATHTOMODEL = 'open-skills-private/model_cache/va_0605/'


class Doc2Vectorizer(object):
    def __init__(self, model_name=None, path=None, s3_conn=None):
        self.model_name = model_name or MODEL_NAME
        self.path = path or PATHTOMODEL
        self.nlp = NLPTransforms()
        self.s3_conn = s3_conn
        self.model = self._load_model(modelname=self.model_name)

    def _load_model(self, modelname):
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')
        filepath = 'tmp/' + modelname
        s3path = self.path + self.model_name
        if not os.path.exists(filepath):
            logging.warning('calling download from %s to %s', s3path, filepath)
            download(self.s3_conn, filepath, s3path)
        else:
            logging.warning('model existed in tmp/')
        return Doc2Vec.load(filepath)

    def vectorize(self, documents):
        for document in documents:
            yield self.model.infer_vector(document)
