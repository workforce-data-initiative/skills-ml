import os
import logging
import json

from gensim.models import Doc2Vec

from skills_ml.algorithms.string_cleaners import NLPTransforms

from skills_utils.s3 import download

PATHTOMODEL = 'open-skills-private/model_cache/'
LOOKUP = 'lookup_va_0605.json'

MODEL_NAME = 'gensim_doc2vec_va_0605'
DOCTAG = 'gensim_doc2vec_va_0605.docvecs.doctag_syn0.npy'
SYN0 = 'gensim_doc2vec_va_0605.syn0.npy'
SYN1 = 'gensim_doc2vec_va_0605.syn1.npy'

class SocClassifier(object):
    def __init__(self, mode=None, model_name=MODEL_NAME, path=PATHTOMODEL, s3_conn=None):
        self.model_name = model_name
        self.path = path
        self.s3_conn = s3_conn
        self.mode = mode
        self.files  = [MODEL_NAME, DOCTAG, SYN0, SYN1]
        self.model = self._load_model()
        self.lookup = self._load_lookup()

    def _load_model(self):
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')

        for f in self.files:
            filepath = 'tmp/' + f
            if not os.path.exists(filepath):
                logging.warning('calling download from %s to %s', self.path + f, filepath)
                download(self.s3_conn, filepath, self.path + f)

        return Doc2Vec.load('tmp/' + self.model_name)

    def _load_lookup(self):
        filepath = 'tmp/' + LOOKUP
        download(self.s3_conn, filepath , PATHTOMODEL + LOOKUP)
        with open(filepath, 'r') as handle:
            lookup = json.load(handle)
        return lookup

    def classify(jobposting):
        pass
