import random
from skills_ml.algorithms.string_cleaners import NLPTransforms
from skills_ml.algorithms.sampling.methods import reservoir, reservoir_weighted
import numpy as np
from skills_ml.utils import safe_get


class JobSampler(object):
    def __init__(self, job_posting_generator, major_group=True, keys=None, weights=None, random_state=None):
        self.job_posting_generator = job_posting_generator
        self.nlp = NLPTransforms()
        self.major_group = major_group
        self.weights = weights
        self.keys = keys
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)

    def _transform_generator(self, job_posting_generator):
        if isinstance(self.keys, list):
            for job in job_posting_generator:
                yield (job, safe_get(job, *self.keys))
        elif self.major_group:
            for job in job_posting_generator:
                 yield (job, job['onet_soc_code'][:2])
        else:
            for job in job_posting_generator:
                yield (job, job['keys'])


    def sample(self, k):
        if self.weights:
            it = self._transform_generator(self.job_posting_generator)
            return list(reservoir_weighted(it, k, self.weights))
        else:
            it = self.job_posting_generator
            return list(reservoir(it, k))
