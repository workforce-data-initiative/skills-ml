from sklearn import metrics
from descriptors import cachedproperty

import logging
import numpy as np

class ClassificationEvaluator(object):
    def __init__(self, result_generator):
        self.result_generator = result_generator
        if not hasattr(self.result_generator,'target_variable'):
            raise AttributeError("the result_generator should have target_variable property")
        else:
            self.target_variable = self.result_generator.target_variable
            self.labels = self.target_variable.choices
        self.result = np.array(list(self.result_generator))

    @cachedproperty
    def y_pred(self):
        return self.target_variable.encoder.inverse_transform(self.result[:, 0])

    @cachedproperty
    def y_true(self):
        return self.target_variable.encoder.inverse_transform(self.result[:, 1])

    @cachedproperty
    def accuracy(self):
        return metrics.accuracy_score(self.y_true, self.y_pred)

    @cachedproperty
    def precision(self):
        return metrics.precision_score(self.y_true, self.y_pred, labels=self.labels, average=None)

    @cachedproperty
    def recall(self):
        return metrics.recall_score(self.y_true, self.y_pred, labels=self.labels, average=None)

    @cachedproperty
    def f1(self):
        return metrics.f1_score(self.y_true, self.y_pred, labels=self.labels, average=None)

    @cachedproperty
    def confusion_matrix(self):
        return metrics.confusion_matrix(self.y_true, self.y_pred)

    @cachedproperty
    def macro_precision(self):
        return metrics.precision_score(self.y_true, self.y_pred, average='macro')

    @cachedproperty
    def micro_precision(self):
        return metrics.precision_score(self.y_true, self.y_pred, average='micro')

    @cachedproperty
    def macro_recall(self):
        return metrics.recall_score(self.y_true, self.y_pred, average='macro')

    @cachedproperty
    def micro_recall(self):
        return metrics.recall_score(self.y_true, self.y_pred, average='micro')

    @cachedproperty
    def macro_f1(self):
        return metrics.f1_score(self.y_true, self.y_pred, average='macro')

    @cachedproperty
    def micro_f1(self):
        return metrics.f1_score(self.y_true, self.y_pred, average='micro')


class OnetOccupationClassificationEvaluator(ClassificationEvaluator):
    def __init__(self, result_generator):
        super().__init__(result_generator)
        if not hasattr(self.result_generator, 'target_variable'):
            raise AttributeError("the result_generator should have target_variable property")
        else:
            self.target_variable = self.result_generator.target_variable
            self.labels = self.target_variable.choices

    @cachedproperty
    def _result_for_major_group(self):
        y_pred = [p[:2] for p in self.y_pred]
        y_true = [t[:2] for t in self.y_true]
        return y_true, y_pred

    @cachedproperty
    def accuracy_major_group(self):
        if self.target_variable.name == 'full_soc':
            y_true, y_pred = self._result_for_major_group
            return metrics.accuracy_score(y_true, y_pred)

        elif self.target_variable.name == 'major_group':
            return self.accuracy

    @cachedproperty
    def recall_per_major_group(self):
        if self.target_variable.name == 'major_group':
            return self.recall
        elif self.target_variable.name == 'full_soc':
            y_true, y_pred = self._result_for_major_group
            return metrics.recall_score(y_true, y_pred, average=None)

    @cachedproperty
    def precision_per_major_group(self):
        if self.target_variable.name == 'major_group':
            return self.precision
        elif self.target_variable.name == 'full_soc':
            y_true, y_pred = self._result_for_major_group
            return metrics.precision_score(y_true, y_pred, average=None)

    @cachedproperty
    def f1_per_major_group(self):
        if self.target_variable.name == 'major_group':
            return self.f1
        elif self.target_variable.name ==  'full_soc':
            y_true, y_pred = self._result_for_major_group
            return metrics.f1_score(y_true, y_pred, average=None)
