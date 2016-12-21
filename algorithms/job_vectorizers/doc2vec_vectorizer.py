import pandas as pd
import gensim
import json
import re
import os
import boto
from tempfile import TemporaryFile

from utils.nlp import NLPTransforms

from airflow.hooks import S3Hook

from utils.s3 import split_s3_path, load2tmp

MODEL_NAME = 'gensim_doc2vec'
PATHTOMODEL = 'skills-private/model_cache/'

class Doc2Vectorizer(object):
    def __init__(self, model_name, path, s3_conn):
        self.model_name = model_name
        self.path = path
        self.nlp = NLPTransforms()
        self.s3_conn = s3_conn

    def _load_model(self, modelname):
        full_path = self.path + self.model_name
        load2tmp(self.s3_conn, modelname, full_path)
        return gensim.models.Doc2Vec.load("tmp/gensim_doc2vec")

    def vectorize(self, document):
        model = self._load_model(modelname=MODEL_NAME)
        return model.infer_vector(self.nlp.clean_split(document))

    def split_train_test(self):
        pass









