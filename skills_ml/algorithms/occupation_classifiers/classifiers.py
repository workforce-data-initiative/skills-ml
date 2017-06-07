import os
import logging
import json
from collections import Counter

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
    """The SocClassifier Object to classify each jobposting description to O*Net SOC code.

    Example:

    Soc = SocClassifier(s3_conn=s3_conn)
    predicted_soc = Soc.classify(jobposting, mode='top')
    """
    def __init__(self, model_name=MODEL_NAME, path=PATHTOMODEL, s3_conn=None):
        """To initialize the SocClassifier Object, the model and lookup disctionary
        will be downloaded to the tmp/ directory and loaded to the memory.

        Attributes:
            model_name (str): the name of the model to be used.
            path (str): the path of the model on S3.
            s3_conn (:obj: `boto.s3.connection.S3Connection`): the boto object to connect to S3.
            files (:obj: `list` of (str)): model files need to be downloaded/loaded.
            model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model.
            lookup (dict): lookup table for mapping each jobposting index to soc code.
        """
        self.model_name = model_name
        self.path = path
        self.s3_conn = s3_conn
        self.files  = [MODEL_NAME, DOCTAG, SYN0, SYN1]
        self.model = self._load_model()
        self.lookup = self._load_lookup()

    def _load_model(self):
        """The method to download the model from S3 and load to the memory.
        """
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')

        for f in self.files:
            filepath = 'tmp/' + f
            if not os.path.exists(filepath):
                logging.warning('calling download from %s to %s', self.path + f, filepath)
                download(self.s3_conn, filepath, self.path + f)

        return Doc2Vec.load('tmp/' + self.model_name)

    def _load_lookup(self):
        """The method to download the lookup dictionary from S3 and load to the memory.
        """
        filepath = 'tmp/' + LOOKUP
        if not os.path.exists(filepath):
            logging.warning('calling download from %s to %s', self.path + LOOKUP, filepath)
            download(self.s3_conn, filepath , self.path + LOOKUP)

        with open(filepath, 'r') as handle:
            lookup = json.load(handle)
        return lookup

    def classify(self, jobposting, mode='top'):
        """The method to predict the soc code a job posting belongs to.

        Args:
            jobposting (str): a string of cleaned, lower-cased and pre-processed job description context.
            mode (str): a flag of which method to use for classifying.
        """
        inferred_vector = self.model.infer_vector(jobposting.split())
        sims = self.model.docvecs.most_similar([inferred_vector], topn=1)
        resultlist = list(map(lambda l: self.lookup[l], [x[0] for x in sims]))
        if mode == 'top':
            predicted_soc = resultlist[0]

        if mode == 'common':
            predicted_soc = [Counter(r).most_common()[0][0] for r in resultlist]

        return predicted_soc

