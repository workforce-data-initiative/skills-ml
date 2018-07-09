from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from skills_ml.algorithms.string_cleaners.nlp import NLPTransforms
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


class ClassifierTrainer(object):
    """Trains a series of classifiers using the same training set
    Args:
        storage (skills_ml.storage)
        target_variable (str)
        embedding_model (gensim.model)
        k_folds (int)
        n_jobs (int)
    """
    def __init__(self, k_folds, n_jobs, grid_config):
        # self.storage = storage
        # self.target_variable = target_variable
        self.k_folds = k_folds
        self.n_jobs = n_jobs
        self.grid_config = grid_config
        self.cls_cv_result = {}

    def _train(self, matrix_store):
        """Fit a model to a training set. Works on any modeling class that
        is vailable in this package's environment and implements .fit

        Args:

        Returns:

        """
        scores = ['accuracy']
        y =  matrix_store.labels
        for class_path, parameter_config in self.grid_config.items():
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            for score in scores:
                cls_cv = GridSearchCV(cls(), parameter_config, cv=self.k_folds, scoring=f'{score}', n_jobs=self.n_jobs)

                self.cls_cv_result[class_name] = {score: cls_cv.fit(matrix_store.matrix, y)}

    def save_result(self):
        return NotImplementedError


def major_group_label_encode(soc):
    onet_soc = list(onetdict.keys())
    label_encoder = LabelEncoder()
    label_encoder.fit(onet_soc)

    return label_encoder.transform([soc[:2]])


def create_training_set(job_postings_generator, embedding_model, document_schema_fields=['description','experienceRequirements', 'qualifications', 'skills']):
    for job in job_postings_generator:
        if job['onet_soc_code'][:2] != '99' and job['onet_soc_code'][:2] != '' :
            label = major_group_label_encode(job['onet_soc_code'])
            text = ' '.join([job[d] for d in document_schema_fields])
            tokens = NLPTransforms().word_tokenize(text)
            yield (embedding_model.infer_vector(tokens), label)


