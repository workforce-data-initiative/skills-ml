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
    def __init__(self, model_name=MODEL_NAME, path=PATHTOMODEL):
        self.model_name = model_name
        self.path = path
        self.nlp = NLPTransforms()

    def _load_model(self):
        full_path = self.path + self.model_name
        s3_conn = S3Hook().get_conn()
        modelname = "doc2vec.model"
        #bucket_name, prefix = split_s3_path(full_path)
        #bucket = s3_conn.get_bucket(bucket_name)
        #key = boto.s3.key.Key(
        #    bucket=bucket,
        #    name=prefix
        #)

        #if not os.path.exists("../../tmp"):
        #    os.makedirs("../../tmp")
        #key.get_contents_to_filename("tmp/doc2vec.model")
        load2tmp(s3_conn, modelname, full_path)
        return gensim.models.Doc2Vec.load("tmp/doc2vec.model")

    def vectorize(self, document):
        model = self._load_model()
        return model.infer_vector(self.nlp.clean_split(document))







