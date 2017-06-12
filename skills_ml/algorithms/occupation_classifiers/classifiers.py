import os
import logging
import json
from collections import Counter, defaultdict

from gensim.models import Doc2Vec

from sklearn import neighbors

from skills_utils.s3 import download, split_s3_path, list_files

from skills_ml.algorithms.occupation_classifiers.base import VectorModel
