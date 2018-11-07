"""Sample job postings"""

import random
from skills_ml.algorithms.sampling.methods import reservoir, reservoir_weighted
import numpy as np
from skills_utils.common import safe_get


class JobSampler(object):
    """Job posting sampler using reservoir sampling methods

    It takes a job_posting generator as an input. To sample based on weights, one should sepecify a weight dictionary.

    Attributes:
        job_posting_generator (iterator): Job posting iterator to sample from.
        k (int): number of documents to sample
        weights (dict): a dictionary that has key-value pairs as label-weighting pairs. It expects every
                        label in the iterator to be present as a key in the weights dictionary For example,
                        weights = {'11': 2, '13', 1}. In this case, the label/key is the occupation major
                        group and the value is the weight you want to sample with.
        key (callable): a function to be called on each element to associate to the key of weights dictionary
        random_state (int): the seed used by the random number generator

    """
    def __init__(self, job_posting_generator, k, weights=None, key=lambda x: x, random_state=None):
        self.job_posting_generator = job_posting_generator
        self.k = k
        self.key = key
        self.weights = weights
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)

    def __iter__(self):
        if self.weights:
            yield from reservoir_weighted(self.job_posting_generator, self.k, self.weights, self.key)
        else:
            yield from reservoir(self.job_posting_generator, self.k)
