from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold

from skills_ml.storage import FSStore
from skills_ml.algorithms.string_cleaners.nlp import NLPTransforms
from skills_ml.ontologies.onet import build_onet, majorgroupname

import numpy as np
import importlib


def get_all_soc(onet=None):
    if not onet:
        onet = build_onet()
    occupations = onet.occupations
    soc = []
    for occ in occupations:
        if 'O*NET-SOC Occupation' in occ.other_attributes['categories']:
            soc.append(occ.identifier)

    return soc

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
        self.target_variable = self.matrix.target_variable
        self.k_folds = k_folds
        self.n_jobs = n_jobs
        self.grid_config = self.default_grid_config if grid_config is None else grid_config
        self.cls_cv_result = {}
        self.scoring = scoring
        self.best_classifiers = {}
        self.random_state_for_split = random_state_for_split

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

    def train(self):
        """Fit a model to a training set. Works on any modeling class that
        is vailable in this package's environment and implements .fit
        """
        X = self.matrix.X
        y = self.matrix.y
        for score in self.scoring:
            for class_path, parameter_config in self.grid_config.items():
                module_name, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                kf = StratifiedKFold(n_splits=self.k_folds, random_state=self.random_state_for_split)

                cls_cv = GridSearchCV(cls(), parameter_config, cv=kf, scoring=score, n_jobs=self.n_jobs)
                cls_cv.fit(X, y)
                self.cls_cv_result[score] = {class_name: cls_cv.cv_results_}
                self.best_classifiers[score] = {class_name: cls_cv}

    def save_result(self):
        return NotImplementedError


class SocEncoder(LabelEncoder):
    def __init__(self, label_list):
        self.fit(label_list)


def create_training_set(job_postings_generator, embedding_model=None, target_variable="major_group",
                        document_schema_fields=['description','experienceRequirements', 'qualifications', 'skills']):
    """Create training set for occupation classifier from job postings generator and embedding model

    Args:
        job_postings_generator (generator): job posting collection
        emebdding_model (skills_ml.algorithm.embedding.models): embedding model
        document_schema_fields (list): fields to be included in the training data

    Returns:
        (dict) a dictionary of training data(X), label(y), the embedding_model, target_variable and soc_encoder
    """
    X = []
    y = []

    if target_variable == "major_group":
        se = SocEncoder(list(majorgroupname.keys()))
        label_transformer = lambda soc_code: se.transform([soc_code[:2]])
    elif target_variable == "full_soc":
        se = SocEncoder(get_all_soc())
        label_transformer = lambda soc_code: se.transform([soc_code])

    for job in job_postings_generator:
        if job['onet_soc_code'][:2] != '99' and job['onet_soc_code'] != '' :
            label = label_transformer(job['onet_soc_code'])

            text = ' '.join([NLPTransforms().clean_str(job[field]) for field in document_schema_fields])
            tokens = NLPTransforms().word_tokenize(text)
            if embedding_model:
                X.append(embedding_model.infer_vector(tokens))
            else:
                X.append(tokens)
            y.append(label)

    return Matrix(X, y, embedding_model, target_variable, se)


class Matrix(object):
    def __init__(self, X, y, embedding_model, target_variable, soc_encoder):
        self._X = X
        self._y = y
        self.embedding_model = embedding_model
        self.target_variable = target_variable
        self.soc_encoder = soc_encoder

    @property
    def X(self):
        return np.array(self._X)

    @property
    def y(self):
        return np.reshape(np.array(self._y), len(self._y), 1)
