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

    @property
    def _all_filters(self):
        if self.filters:
            if isinstance(self.filters, list):
                return self.default_filters + self.filters
            else:
                return self.default_filters + [self.filters]
        else:
            return self.default_filters

    @property
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


class SOCMajorGroup(TargetVariable):
    name = 'major_group'

    def __init__(self, filters=None):
        super().__init__(filters)
        self.default_filters = [unknown_soc_filter, empty_soc_filter]
        self.choices = list(majorgroupname.keys())
        self.encoder = SocEncoder(self.choices)

    @property
    def transformer(self):
        return lambda job_posting: self.encoder.transform([get_onet_occupation(job_posting)[:2]])


class FullSOC(TargetVariable):
    name = 'full_soc'

    def __init__(self, filters=None, onet_cache=None):
        super().__init__(filters)
        self.default_filters = [unknown_soc_filter, empty_soc_filter]
        self.onet = Onet(onet_cache)
        self.choices = self.onet.all_soc
        self.encoder = SocEncoder(self.choices)

    @property
    def transformer(self):
        return lambda job_posting: self.encoder.transform([get_onet_occupation(job_posting)])


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
        combined = zip_longest(self.pipe_X.build(gen1), self.pipe_y.build(gen2))
        return combined

    def build(self):
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
        meta_dict.update(self.data_source_generator.metadata)
        return meta_dict
