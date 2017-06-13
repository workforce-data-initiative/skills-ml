import os
import logging
import json
from collections import Counter, defaultdict

from skills_ml.algorithms.occupation_classifiers.nearestneighbor import NearestNeighbors

class Classifiers(object):
    def __init__(self, classifier_name=None, s3_conn=None):
        self.classifier_list = ['ann', 'knn', 'svm', 'softmax']
        self.classifier_name = classifier_name
        self.s3_conn = s3_conn
        self.classifier = self._load_classifier()


    def _load_classifier(self):
        if self.classifier_name == None:
            return NearestNeighbors(s3_conn=self.s3_conn)
        else:
            print('Not implemented yet!')
            return None


    def predict_soc(self, jobposting, **kwargs):
        return self.classifier.classify(jobposting, **kwargs)


