import random
from skills_ml.algorithms.sampling.methods import reservoir, reservoir_weighted

class JobSampler(object):
    def __init__(self, job_posting, lookup, weights=None, random_state=None):
        self.job_posting = job_posting
        self.lookup = lookup
        self.weights = weights
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)

    def _transform_iterator(self, job_iter, lookup):
        for job in job_iter:
            yield (job[0], lookup[job[1][0]])

    def sample(self, k):
        it = self._transform_iterator(self.job_posting, self.lookup)
        if self.weights:
            return list(reservoir_weighted(it, k, self.weights))
        else:
            return list(reservoir(it, k))
