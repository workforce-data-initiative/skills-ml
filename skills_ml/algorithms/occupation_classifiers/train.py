from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.externals import joblib

from skills_ml.storage import FSStore, open_sesame
from skills_ml.ontologies.onet import majorgroupname
from skills_ml.algorithms.nlp import clean_str, word_tokenize
from skills_ml.algorithms.occupation_classifiers import SocEncoder, SOCMajorGroup, TargetVariable, DesignMatrix
from skills_ml.algorithms.preprocessing import IterablePipeline
from skills_ml.job_postings.common_schema import JobPostingGeneratorType
from skills_ml.utils import filename_friendly_hash
from skills_ml.storage import ModelStorage, ProxyObjectWithStorage

import importlib
import logging
from datetime import datetime
from itertools import zip_longest, tee
from typing import Type, Union
import dill
import inspect
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
        self.random_state_for_split = random_state_for_split
        self.train_time = datetime.today().isoformat()
        self.best_estimators = []

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
        logging.info(f"Start training {self.train_time}")
        if len(self.matrix.X) == 0:
            self.matrix.build()

        X = self.matrix.X
        y = self.matrix.y
        store_path = os.path.join(self.storage.path, self.train_time)
        for score in self.scoring:
            self.cls_cv_result[score] = {}
            for class_path, parameter_config in self.grid_config.items():
                module_name, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)

                logging.info(f"training {class_name}")
                kf = StratifiedKFold(n_splits=self.k_folds, random_state=self.random_state_for_split)
                model_hash = self._model_hash(self.matrix.metadata, class_name, parameter_config)
                trained_model_name = class_name.lower() + "_" + model_hash
                self.storage.path = os.path.join(store_path, score)
                if 'n_jobs' in inspect.signature(cls).parameters.keys():
                    cls_cv = ProxyObjectWithStorage(
                            model_obj=GridSearchCV(
                                estimator=cls(n_jobs=self.n_jobs),
                                param_grid=parameter_config,
                                cv=kf, scoring=score),
                            storage=self.storage,
                            model_name=trained_model_name,
                            target_variable=self.matrix.target_variable
                            )
                else:
                    cls_cv = ProxyObjectWithStorage(
                            model_obj=GridSearchCV(
                                estimator=cls(),
                                param_grid=parameter_config,
                                cv=kf,
                                scoring=score),
                            storage=self.storage,
                            model_name=trained_model_name,
                            target_variable=self.matrix.target_variable
                            )
                cls_cv.fit(X, y)
                self.cls_cv_result[score][class_name] = cls_cv.cv_results_
                if save:
                    logging.info(f"storing {class_name} {model_hash} to {store_path}")
                    self._save(cls_cv, os.path.join(store_path, score, trained_model_name))
                else:
                    logging.info("appending classifier")
                    self.best_estimators.append(cls_cv)

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
        with open_sesame(path_to_save, 'wb') as f:
            dill.dump(cls_cv, f)

