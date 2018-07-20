from sklearn.preprocessing import LabelEncoder

from skills_ml.ontologies.onet import majorgroupname

from abc import ABC, abstractmethod
import numpy as np

def get_all_soc(onet=None):
    if not onet:
        onet = build_onet()
    occupations = onet.occupations
    soc = []
    for occ in occupations:
        if 'O*NET-SOC Occupation' in occ.other_attributes['categories']:
            soc.append(occ.identifier)

    return soc


class SocEncoder(LabelEncoder):
    def __init__(self, label_list):
        self.fit(label_list)


unknown_soc_filter = lambda job: job['onet_soc_code'][:2] != '99'
empty_soc_filter = lambda job: job['onet_soc_code'] != ''


class TargetVariable(ABC):
    def __init__(self, filters=None):
        self.default_filters = [unknown_soc_filter, empty_soc_filter]
        self.filters = filters
        self.filter_func =  lambda x: all(f(x) for f in self.all_filters)

    @property
    def all_filters(self):
        if self.filters:
            if isinstance(self.filters, list):
                return self.default_filters + self.filters
            else:
                return self.default_filters + [self.filters]
        else:
            return self.default_filters

    @property
    @abstractmethod
    def name(self):
        pass


class SOCMajorGroup(TargetVariable):
    name = 'major_group'

    def __init__(self, filters=None):
        super().__init__(filters)
        self.choices = list(majorgroupname.keys())
        self.encoder = SocEncoder(self.choices)
        self.transformer = lambda job_posting: self.encoder.transform([job_posting['onet_soc_code'][:2]])


class TrainingMatrix(object):
    def __init__(self, X, y, embedding_model, target_variable):
        self._X = X
        self._y = y
        self.embedding_model = embedding_model
        self.target_variable = target_variable

    @property
    def X(self):
        return np.array(self._X)

    @property
    def y(self):
        return np.reshape(np.array(self._y), len(self._y), 1)
