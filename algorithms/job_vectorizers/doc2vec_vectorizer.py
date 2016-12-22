import pandas as pd
import json
import os
from gensim.models import Doc2Vec

from utils.nlp import NLPTransforms

from utils.s3 import split_s3_path, load2tmp

MODEL_NAME = 'gensim_doc2vec'
PATHTOMODEL = 'skills-private/model_cache/'

class Doc2Vectorizer(object):
    def __init__(self, model_name=MODEL_NAME, path=PATHTOMODEL, s3_conn=None):
        self.model_name = model_name
        self.path = path
        self.nlp = NLPTransforms()
        self.s3_conn = s3_conn

    def _load_model(self, modelname):
        full_path = self.path + self.model_name
        if not os.path.exists("tmp/gensim_doc2vec"):
            load2tmp(self.s3_conn, modelname, full_path)
        return Doc2Vec.load("tmp/gensim_doc2vec")

    def vectorize(self, documents):
        model = self._load_model(modelname=MODEL_NAME)
        for document in documents:
            #print('vectorizing...')
            yield model.infer_vector(document)

    def split_train_test(self):
        pass









