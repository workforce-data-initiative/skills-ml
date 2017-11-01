import random
from skills_ml.algorithms.sampling.methods import reservoir, reservoir_weighted
import numpy as np

class JobSampler(object):
    def __init__(self, job_posting_iter, lookup=None, weights=None, random_state=None):
        self.job_posting_iter = job_posting_iter
        self.lookup = lookup
        self.weights = weights
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)

    def _transform_iterator(self, job_posting_iter):
        if self.lookup:
            for job in job_posting_iter:
                yield (job[0], lookup[job[1][0]])
        else:
            for job in job_posting_iter:
                yield (job, )


    def sample(self, k):
        if type(self.job_posting_iter).__name__ == 'CorpusCreator':
            it = self.job_posting_iter

        elif type(self.job_posting_iter).__name__ == 'Doc2VecGensimCorpusCreator' and self.lookup:
            it = self._transform_iterator(self.job_posting_iter, self.lookup)

        elif type(self.job_posting_iter).__name__ == 'Word2VecGensimCorpusCreator' and self.lookup:
            it = self._transform_iterator(self.job_posting_iter, self.lookup)

        else:
            raise("Please specify the right corpus and lookup table")


        if self.weights:
            return list(reservoir_weighted(it, k, self.weights))

        else:
            return list(reservoir(it, k))
