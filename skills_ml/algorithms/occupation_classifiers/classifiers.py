import os
import logging
from gensim.models import Doc2Vec

from skills_ml.algorithms.string_cleaners import NLPTransforms

from skills_utils.s3 import download

MODEL_NAME = 'gensim_doc2vec'
PATHTOMODEL = 'skills-private/model_cache/'

