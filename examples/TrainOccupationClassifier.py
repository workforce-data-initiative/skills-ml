from skills_ml.storage import ModelStorage, FSStore

from skills_ml.job_postings.common_schema import  JobPostingCollectionSample
from skills_ml.job_postings.filtering import JobPostingFilterer

from skills_ml.algorithms.embedding.models import Word2VecModel
from skills_ml.algorithms.preprocessing import IterablePipeline
from skills_ml.algorithms import nlp
from skills_ml.algorithms.occupation_classifiers.train import OccupationClassifierTrainer
from skills_ml.algorithms.occupation_classifiers import FullSOC, DesignMatrix

import os
import json
import random
from functools import partial
import logging
logging.basicConfig(level=logging.INFO, filename=os.path.abspath("grid_search.log"))
logging.getLogger().addHandler(logging.StreamHandler())
import multiprocessing
num_of_worker = multiprocessing.cpu_count()


job_samples = JobPostingCollectionSample()
job_postings = list(job_samples)

random.shuffle(job_postings)

train_data = job_postings[:30]
test_data = job_postings[30:]

train_bytes = json.dumps(train_data).encode()
test_bytes = json.dumps(test_data).encode()


logging.info("Loading Embedding Model")
model_storage = ModelStorage(FSStore('/your/model/path'))
w2v = model_storage.load_model(model_name='your_model_name')

full_soc = FullSOC()

def basic_filter(doc):
    """
    Return the document except for the document which soc is unknown or empty or not in the
    soc code pool of current O*Net version
    """
    if full_soc.filter_func(doc) and doc['onet_soc_code'] in full_soc.choices:
        return doc
    else:
        return None

class JobGenerator(object):
    def __init__(self, data):
        self.data = data

    @property
    def metadata(self):
        return job_samples.metadata

    def __iter__(self):
        yield from self.data

document_schema_fields = ['description', 'experienceRequirements', 'qualifications', 'skills']
pipe_x = IterablePipeline(
    basic_filter,
    partial(nlp.fields_join, document_schema_fields=document_schema_fields),
    nlp.clean_str,
    nlp.word_tokenize,
    partial(nlp.vectorize, embedding_model=w2v)
)
pipe_y = IterablePipeline(
    basic_filter,
    full_soc.transformer
)

matrix = DesignMatrix(
        data_source_generator=JobGenerator(train_data),
        target_variable=full_soc,
        pipe_X=pipe_x,
        pipe_y=pipe_y)

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
    storage=FSStore('tmp/soc_classifiers'),
    n_jobs = num_of_worker
)
trainer.train()

fs = FSStore(os.path.join('soc_classifiers', trainer.train_time))
fs.write(train_bytes, "train.data")
fs.write(test_bytes, "test_data")

