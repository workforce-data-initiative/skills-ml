from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.externals import joblib

from skills_ml.storage import FSStore
from skills_ml.ontologies.onet import majorgroupname
from skills_ml.algorithms.string_cleaners.nlp import clean_str, word_tokenize
from skills_ml.algorithms.occupation_classifiers import SocEncoder, SOCMajorGroup, TargetVariable, TrainingMatrix
from skills_ml.algorithms.preprocessing import IterablePipeline
from skills_ml.algorithms.embedding.base import ModelStorage
from skills_ml.job_postings.common_schema import JobPostingGeneratorType
from skills_ml.utils import filename_friendly_hash

import importlib
import logging
from datetime import datetime
from itertools import zip_longest, tee
from typing import Type, Union
import pickle
import os

class OccupationClassifierTrainer(object):
    """Trains a series of classifiers using the same training set
    Args:
        matrix (skills_ml.algorithms.train.matrix): a matrix object holds X, y and other training data information
        storage (skills_ml.storage): a skills_ml storage object specified the store method
        k_folds (int): number of folds for cross validation
        random_state_for_split(int): random state
        n_jobs (int): umber of jobs to run in parallel
        scores:
    """
    def __init__(self, matrix, k_folds, grid_config=None, storage=None,
                 random_state_for_split=None, scoring=['accuracy'], n_jobs=3):
        self.matrix = matrix
        self.storage = FSStore() if storage is None else storage
        self.k_folds = k_folds
        self.n_jobs = n_jobs
        self.grid_config = self.default_grid_config if grid_config is None else grid_config
        self.cls_cv_result = {}
        self.scoring = scoring
        self.best_classifiers = {}
        self.random_state_for_split = random_state_for_split
        self.train_time = datetime.today().isoformat()

    @property
    def default_grid_config(self):
        return {
                'sklearn.ensemble.ExtraTreesClassifier': {
                    'n_estimators': [100, 500],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [1, 5, 10, 20, 50],
                    'max_features': ['sqrt', 'log2'],
                    'min_samples_split': [2, 5, 10]
                    },
                'sklearn.ensemble.RandomForestClassifier': {
                    'n_estimators': [100, 500],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [1, 5, 10, 20, 50],
                    'max_features': ['sqrt', 'log2'],
                    'min_samples_split': [2, 5, 10]
                    }
                }

    def train(self, save=True):
        """Fit a model to a training set. Works on any modeling class that
        is vailable in this package's environment and implements .fit
        """
        X = self.matrix.X
        y = self.matrix.y
        store_path = os.path.join(self.storage.path, self.train_time)
        for score in self.scoring:
            self.cls_cv_result[score] = {}
            # self.best_classifiers[score] = {}
            for class_path, parameter_config in self.grid_config.items():
                module_name, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                logging.info(f"training {class_name}")
                kf = StratifiedKFold(n_splits=self.k_folds, random_state=self.random_state_for_split)
                cls_cv = GridSearchCV(cls(), parameter_config, cv=kf, scoring=score, n_jobs=self.n_jobs)
                cls_cv.fit(X, y)
                self.cls_cv_result[score][class_name] = cls_cv.cv_results_
                # self.best_classifiers[score][class_name] = cls_cv
                if save:
                    model_hash = self._model_hash(self.matrix.metadata, class_name, cls_cv.best_params_)
                    logging.info(f"storing {class_name} {model_hash} to {store_path}")
                    self._save(cls_cv, os.path.join(store_path, score, model_hash))

    def unique_parameters(self, parameters):
        return {
            key: parameters[key]
            for key in parameters.keys()
            if key != 'n_jobs'
        }

    def _model_hash(self, matrix_metadata, class_path, parameters):
        unique = {
            'className': class_path,
            'parameters': self.unique_parameters(parameters),
            'project_path': self.storage.path,
            'training_metadata': matrix_metadata
        }
        logging.info(f'Creating model hash from unique data {unique}')
        return filename_friendly_hash(unique)

    def _save(self, cls_cv, path_to_save):
        # store_path = os.path.join(self.storage.path, self.train_time)
        # for score, cls_dict in self.best_classifiers.items():
            # for class_name, cls_cv in cls_dict.items():
                # model_hash = self._model_hash(self.matrix.metadata, class_name, cls_cv.best_classifiers)
        with self.storage.open(path_to_save, 'wb') as f:
            joblib.dump(cls_cv, f, compress=True)


def create_training_set(job_postings_generator: JobPostingGeneratorType,
                        target_variable: TargetVariable,
                        pipe_x: IterablePipeline=None,
                        pipe_y: IterablePipeline=None,
                        ) -> TrainingMatrix:
    """Create training set for occupation classifier from job postings generator and embedding model

    Args:
        job_postings_generator (generator): job posting collection
        emebdding_model (skills_ml.algorithm.embedding.models): embedding model
        document_schema_fields (list): fields to be included in the training data

    Returns:
        (skills_ml.algorithm.occupation_classifiers.TrainingMatrix) a matrix class with properties of training data(X),
        label(y), the embedding_model, target_variable and soc_encoder
    """
    X = []
    y = []
    jp1, jp2 = tee(job_postings_generator, 2)
    pxy = zip_longest(pipe_x.build(jp1), pipe_y.build(jp2))
    for i, item in enumerate(pxy):
        X.append(item[0])
        y.append(item[1])


    total = i + 1
    filtered = len(y)
    dropped = 1 - float(len(y) / (i+1))
    logging.info(f"total jobpostings: {total}")
    logging.info(f"filtered jobpostings: {filtered}")
    logging.info(f"dropped jobposting: {dropped}")
    return TrainingMatrix(X, y, pipe_x, pipe_y, job_postings_generator.metadata)
