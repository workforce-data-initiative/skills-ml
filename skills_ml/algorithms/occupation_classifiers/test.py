from skills_ml.algorithms.preprocessing import IterablePipeline
from skills_ml.algorithms.occupation_classifiers.classifiers import CombinedClassifier
from skills_ml.algorithms.string_cleaners import nlp
from skills_ml.job_postings.common_schema import JobPostingGeneratorType

from typing import List, Callable
from functools import partial
from itertools import tee

class OccupationClassifierTester(object):
    def __init__(self,
            test_data_generator: JobPostingGeneratorType,
            preprocessing: List[Callable],
            classifier: CombinedClassifier):
        self.test_data_generator = test_data_generator
        self.classifier = classifier
        self.target_variable = self.classifier.target_variable
        self._preprocessing_X= preprocessing
        self._preprocessing_y = [self.target_variable.transformer]
        self.counter = 0
        self.pipeline_X = None
        self.pipeline_y = None
        self._build_pipeline()

    def _build_pipeline(self):
        steps = self._preprocessing_X.copy()
        steps.append(lambda x: self.classifier.predict([x]))
        self.pipeline_X = IterablePipeline(*steps)
        self.pipeline_y = IterablePipeline(*self._preprocessing_y)

    def create_test_generator(self):
        for d in self.test_data_generator:
            if self.target_variable.filter(d):
                yield d
            else:
                raise AttributeError("Not valid onet soc code")

    def __iter__(self):
        test_X, test_y = tee(self.create_test_generator())
        for predict, label in zip(self.pipeline_X.build(test_X), self.pipeline_y.build(test_y)):
            yield [predict[0], label[0]]
            self.counter += 1

    def __len__(self):
        return self.counter
