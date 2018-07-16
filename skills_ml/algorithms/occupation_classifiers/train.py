from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold

from skills_ml.storage import FSStore
from skills_ml.algorithms.string_cleaners.nlp import NLPTransforms

import numpy as np
import importlib

onetdict ={'11': 'Management Occupations',
           '13': 'Business and Financial Operations Occupations',
           '15': 'Computer and Mathematical Occupations',
           '17': 'Architecture and Engineering Occupations',
           '19': 'Life, Physical, and Social Science Occupations',
           '21': 'Community and Social Service Occupations',
           '23': 'Legal Occupations',
           '25': 'Education, Training, and Library Occupations',
           '27': 'Arts, Design, Entertainment, Sports, and Media Occupations',
           '29': 'Healthcare Practitioners and Technical Occupations',
           '31': 'Healthcare Support Occupations',
           '33': 'Protective Service Occupations',
           '35': 'Food Preparation and Serving Related Occupations',
           '37': 'Building and Grounds Cleaning and Maintenance',
           '39': 'Personal Care and Service Occupations',
           '41': 'Sales and Related Occupations',
           '43': 'Office and Administrative Support Occupations',
           '45': 'Farming, Fishing, and Forestry Occupations',
           '47': 'Construction and Extraction Occupations',
           '49': 'Installation, Maintenance, and Repair Occupations',
           '51': 'Production Occupations',
           '53': 'Transportation and Material Moving Occupations',
           '55': 'Military Specific Occupations'}


class OccupationClassifierTrainer(object):
    """Trains a series of classifiers using the same training set
    Args:
        storage (skills_ml.storage)
        target_variable (str)
        embedding_model (gensim.model)
        k_folds (int)
        n_jobs (int)
    """
    def __init__(self, k_folds, grid_config=None, target_variable='major_group', storage=None,
                 random_state_for_split=None, scores=['accuracy'], n_jobs=3):
        self.storage = FSStore() if storage is None else storage
        self.target_variable = target_variable
        self.k_folds = k_folds
        self.n_jobs = n_jobs
        self.grid_config = self.default_grid_config if grid_config is None else grid_config
        self.cls_cv_result = {}
        self.scores = scores
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

    def fit(self, X, y):
        """Fit a model to a training set. Works on any modeling class that
        is vailable in this package's environment and implements .fit

        Args:
            X (list or np.array) training vectors
            y (list or np.array) Target relative to X for classification

        """
        for class_path, parameter_config in self.grid_config.items():
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            kf = StratifiedKFold(n_splits=self.k_folds, random_state=self.random_state_for_split)
            for score in self.scores:
                cls_cv = GridSearchCV(cls(), parameter_config, cv=kf, scoring=score, n_jobs=self.n_jobs)

                self.cls_cv_result[class_name] = {score: cls_cv.fit(X, y)}

    def save_result(self):
        return NotImplementedError


class SocEncoder(LabelEncoder):
    def __init__(self):
        onet_soc = list(onetdict.keys())
        self.fit(onet_soc)


def create_training_set(job_postings_generator, embedding_model=None, document_schema_fields=['description','experienceRequirements', 'qualifications', 'skills']):
    X = []
    y =[]
    for job in job_postings_generator:
        if job['onet_soc_code'][:2] != '99' and job['onet_soc_code'][:2] != '' :
            se = SocEncoder()
            label = se.transform([job['onet_soc_code'][:2]])
            text = ' '.join([job[d] for d in document_schema_fields])
            tokens = NLPTransforms().word_tokenize(text)
            if embedding_model:
                X.append(embedding_model.infer_vector(tokens))
            else:
                X.append(tokens)
            y.append(label)
    return np.array(X), np.reshape(np.array(y), len(y), 1)


