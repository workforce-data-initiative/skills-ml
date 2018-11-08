from sklearn.preprocessing import LabelEncoder

from skills_ml.algorithms.preprocessing import IterablePipeline
from skills_ml.ontologies.onet import Onet, majorgroupname
from skills_ml.job_postings.common_schema import get_onet_occupation

from abc import ABC, abstractmethod
import numpy as np
from itertools import zip_longest, tee
from typing import Generator
import logging


class SocEncoder(LabelEncoder):
    def __init__(self, label_list):
        self.fit(label_list)

unknown_soc_filter = lambda job: job['onet_soc_code'][:2] != '99'
empty_soc_filter = lambda job: job['onet_soc_code'] != ''


class TargetVariable(ABC):
    def __init__(self, filters=None):
        self.default_filters = []
        self.filters = filters
        self.filter_func =  lambda x: all(f(x) for f in self._all_filters)
        self.ontology = None

    @property
    def _all_filters(self):
        if self.filters:
            if isinstance(self.filters, list):
                return self.default_filters + self.filters
            else:
                return self.default_filters + [self.filters]
        else:
            return self.default_filters

    def filter(self, item):
        if self.filter_func(item):
            return item

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def transformer(self):
        pass

    @abstractmethod
    def extract_occupation_from_jobposting(self, job_posting):
        pass


class SOCMajorGroup(TargetVariable):
    name = 'major_group'

    def __init__(self, filters=None):
        super().__init__(filters)
        self.default_filters = [unknown_soc_filter, empty_soc_filter]
        self.choices = list(majorgroupname.keys())
        self.encoder = SocEncoder(self.choices)

    @property
    def transformer(self):
        return lambda job_posting: self.encoder.transform([self.extract_occupation_from_jobposting(job_posting)[0]])

    def extract_occupation_from_jobposting(self, job_posting):
        return (get_onet_occupation(job_posting)[:2], job_posting['id'])


class FullSOC(TargetVariable):
    name = 'full_soc'

    def __init__(self, filters=None, onet_cache=None):
        super().__init__(filters)
        self.default_filters = [unknown_soc_filter, empty_soc_filter]
        self.choices = onet_cache.all_soc if onet_cache else Onet().all_soc
        self.encoder = SocEncoder(self.choices)

    def extract_occupation_from_jobposting(self, job_posting):
        return (get_onet_occupation(job_posting), job_posting['id'])

    @property
    def transformer(self):
        return lambda job_posting: self.encoder.transform([self.extract_occupation_from_jobposting(job_posting)[0]])


class DesignMatrix(object):
    def __init__(
            self,
            data_source_generator: Generator,
            target_variable: TargetVariable,
            pipe_X: IterablePipeline=None,
            pipe_y: IterablePipeline=None):

        if pipe_X == None:
            pipe_X = IterablePipeline()
        if pipe_y == None:
            pipe_y = IterablePipeline()
        if not self._check_pipeline(pipe_X) or not self._check_pipeline(pipe_y):
            raise TypeError("pipeline object should be IterablePipeline object")

        self._X = []
        self._y = []
        self.data_source_generator = data_source_generator
        self.pipe_X = pipe_X
        self.pipe_y = pipe_y
        self.target_variable = target_variable

    def _check_pipeline(self, pipeline):
        return isinstance(pipeline, IterablePipeline)

    def _combine_pipelines(self):
        gen1, gen2 = tee(self.data_source_generator, 2)
        combined = zip_longest(self.pipe_X(gen1), self.pipe_y(gen2))
        return combined

    def build(self):
        logging.info("Building matrix")
        for i, item in enumerate(self._combine_pipelines()):
            self._X.append(item[0])
            self._y.append(item[1])

        total = i + 1
        filtered = len(self._y)
        dropped = 1 - float(filtered / (i+1))
        logging.info(f"total jobpostings: {total}")
        logging.info(f"filtered jobpostings: {filtered}")
        logging.info(f"dropped jobposting: {dropped}")

    @property
    def X(self):
        return np.array(self._X)

    @property
    def y(self):
        return np.reshape(np.array(self._y), len(self._y), 1)

    @property
    def metadata(self):
        meta_dict = {
                'pipe_X': self.pipe_X.description,
                'pipe_y': self.pipe_y.description,
                'target_variable': self.target_variable.name
                }
        return meta_dict
