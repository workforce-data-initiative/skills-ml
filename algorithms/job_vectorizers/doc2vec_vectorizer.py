import os
import logging
from gensim.models import Doc2Vec

from utils.nlp import NLPTransforms

from utils.s3 import download

MODEL_NAME = 'gensim_doc2vec'
PATHTOMODEL = 'skills-private/model_cache/'


class Doc2Vectorizer(object):
    def __init__(self, model_name=MODEL_NAME, path=PATHTOMODEL, s3_conn=None):
        self.model_name = model_name
        self.path = path
        self.nlp = NLPTransforms()
        self.s3_conn = s3_conn

    def _load_model(self, modelname):
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')
        filepath = 'tmp/' + modelname
        s3path = self.path + self.model_name
        if not os.path.exists(filepath):
            logging.warning('calling download from %s to %s', s3path, filepath)
            download(self.s3_conn, filepath, s3path)
        return Doc2Vec.load(filepath)

    def vectorize(self, documents):
        model = self._load_model(modelname=self.model_name)
        for document in documents:
            yield model.infer_vector(document)
