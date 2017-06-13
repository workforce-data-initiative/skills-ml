import os
import json
from collections import Counter, defaultdict

from skills_ml.algorithms.occupation_classifiers.nearestneighbor import NearestNeighbors

class Classifiers(object):
    """The VectorModel Object to classify each jobposting description to O*Net SOC code.

    Example:

    from airflow.hooks import S3Hook
    from skills_ml.algorithms.occupation_classifiers.base import Classifiers

    s3_conn = S3Hook().get_conn()
    Soc = Classifiers(s3_conn=s3_conn, classifier_name='ann')

    predicted_soc = Soc.classify(jobposting, mode='top')
    """
    def __init__(self, classifier_name=None, s3_conn=None):
        self.classifier_list = ['ann', 'knn', 'svm', 'softmax']
        self.classifier_name = classifier_name
        self.s3_conn = s3_conn
        self.classifier = self._load_classifier()


    def _load_classifier(self):
        if self.classifier_name == 'ann':
            return NearestNeighbors(s3_conn=self.s3_conn, indexed=True)
        elif self.classifier_name == 'knn':
            return NearestNeighbors(s3_conn=self.s3_conn, indexed=False)
        else:
            print('Not implemented yet!')
            return None


    def classify(self, jobposting, **kwargs):
        return self.classifier.predict_soc(jobposting, **kwargs)


