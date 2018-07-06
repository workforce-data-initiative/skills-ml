from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
import importlib

class ClassifierTrainer(object):
    """Trains a series of classifiers using the same training set
    Args:
        storage (skills_ml.storage)
        target_variable (str)
        embedding_model (gensim.model)
        fold (int)
    """

def _train(self, matrix_store):
    """Fit a model to a training set. Works on any modeling class that
    is vailable in this package's environment and implements .fit

    Args:

    Returns:

    """
    scores = ['precision', 'recall']
    y =  matrix_store.labels()
    for class_path, parameter_config in grid_config.items():
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        cls_cv = GridSearchCV(cls, parameter_config, cv=5, scoring=f'{score}_macro')
        return cls_cv.fit(matrix_store.matrix, y)


