import random
from skills_ml.algorithms.string_cleaners import NLPTransforms
from skills_ml.algorithms.sampling.methods import reservoir, reservoir_weighted
import numpy as np
from skills_utils.common import safe_get


class JobSampler(object):
    """Job posting sampler using reservoir sampling methods

    It takes a job_posting generator as an input. To sample based on weights, one should sepecify a weight dictionary.

    Attributes:
        job_posting_generator (iterator): Job posting iterator to sample from.
        major_group (bool): A flag for using major_group as a label or not
        keys (list|str): a key or keys(for nested dictionary) indicates the label which should exist in common schema
                         of job posting.
        weights (dict): a dictionary that has key-value pairs as label-weighting pairs. It expects every
                        label in the iterator to be present as a key in the weights dictionary For example,
                        weights = {'11': 2, '13', 1}. In this case, the label/key is the occupation major
                        group and the value is the weight you want to sample with.
        random_state (int): the seed used by the random number generator

    """
    def __init__(self, job_posting_generator, major_group=False, keys=None, weights=None, random_state=None):
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
        elif isinstance(self.keys, str):
            for job in job_posting_generator:
                yield (job, job[self.keys])
        elif self.major_group:
            for job in job_posting_generator:
                try:
                    yield (job, job['onet_soc_code'][:2])
                except TypeError:
                    yield (job, None)
        else:
            for job in job_posting_generator:
                yield (job, )

    def sample(self, k):
        """ Sample method

        Args:
            k (int): number of documents to sample

        Returns:
            list of sampled documents
        """
        it = self._transform_generator(self.job_posting_generator)
        if self.weights:
            return list(reservoir_weighted(it, k, self.weights))
        else:
            return list(reservoir(it, k))
