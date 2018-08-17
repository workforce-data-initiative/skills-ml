import boto
s3_conn = boto.connect_s3()

from skills_ml.storage import FSStore, S3Store

from skills_ml.job_postings.common_schema import JobPostingCollectionFromS3, JobPostingCollectionSample
from skills_ml.job_postings.filtering import JobPostingFilterer

from skills_ml.algorithms.embedding.models import Word2VecModel
from skills_ml.algorithms.preprocessing import IterablePipeline
from skills_ml.algorithms.string_cleaners import nlp
from skills_ml.algorithms.occupation_classifiers.train import OccupationClassifierTrainer
from skills_ml.algorithms.occupation_classifiers import FullSOC, DesignMatrix

import os
import json
import random
from functools import partial
import logging
logging.basicConfig(level=logging.INFO, filename="/home/ubuntu/tweddielin/skills-ml/grid_search.log")
logging.getLogger().addHandler(logging.StreamHandler())
import multiprocessing
num_of_worker = multiprocessing.cpu_count()

job_24k = JobPostingCollectionFromS3(s3_conn=s3_conn, s3_paths="open-skills-private/sampled_jobpostings/samples_24k_v1")
job_24k = list(job_24k)

random.shuffle(job_24k)

train_data = job_24k[:19200]
test_data = job_24k[19200:]

train_bytes = json.dumps(train_data).encode()
test_bytes = json.dumps(test_data).encode()


logging.info("Downloading Embedding Model")
w2v = Word2VecModel.load(storage=S3Store('open-skills-private/model_cache/embedding'), model_name='word2vec_2018-07-27T20:01:27.895533.model')

full_soc = FullSOC()

def filter1(doc):
    if full_soc.filter_func(doc) and doc['onet_soc_code'] in full_soc.onet.all_soc:
        return doc
    else:
        return None

class JobGenerator(object):
    def __init__(self, data):
        self.data = data

    @property
    def metadata(self):
        return {'job postings': {'downloaded_from': 'open-skills-private/sampled_jobpostings/samples_24k_v1'}}
    def __iter__(self):
        for d in self.data:
            yield d

document_schema_fields = ['description', 'experienceRequirements', 'qualifications', 'skills']
pipe_x = IterablePipeline(
    filter1,
    partial(nlp.fields_join, document_schema_fields=document_schema_fields),
    nlp.clean_str,
    nlp.word_tokenize,
    partial(nlp.vectorize, embedding_model=w2v)
)
pipe_y = IterablePipeline(
    filter1,
    full_soc.transformer
)

matrix = DesignMatrix(JobGenerator(train_data), full_soc, pipe_x, pipe_y)
matrix.build()

grid_config = {
                 'sklearn.ensemble.ExtraTreesClassifier': {
                     'n_estimators': [50, 100, 500, 1000],
                     'criterion': ['entropy'],
                     'max_depth': [20, 50],
                     'max_features': ['log2'],
                     'min_samples_split': [10, 20]
                     },
                 'sklearn.ensemble.RandomForestClassifier': {
                     'n_estimators': [50, 100, 500, 1000],
                     'criterion': ['entropy'],
                     'max_depth': [20, 50],
                     'max_features': ['log2'],
                     'min_samples_split': [10, 20]
                     },
                 'sklearn.neural_network.MLPClassifier': {
                    'hidden_layer_sizes': [100, 200, 300, 500, 1000],
                     'activation': ['identity', 'logistic', 'tanh', 'relu'],
                     'solver': ['lbfgs', 'sgd', 'adam']
                     },
                 'sklearn.svm.SVC': {
                     'C': [0.1, 1, 10, 100, 1000],
                     'kernel': ['linear', 'poly', 'sigmoid', 'rbf', 'precomputed'],
                     'shrinking': [True, False],
                     'decision_function_shape': ['ovo', 'ovr']
                     }
                 }

trainer = OccupationClassifierTrainer(
    matrix=matrix,
    k_folds=3,
    grid_config=grid_config,
    storage=S3Store('open-skills-private/model_cache/soc_classifiers'),
    n_jobs = num_of_worker
)
trainer.train()

s3 = S3Store(os.path.join('open-skills-private/model_cache/soc_classifiers', trainer.train_time))
s3.write(train_bytes, "train.data")
s3.write(test_bytes, "test_data")

