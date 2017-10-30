import random
from skills_ml.algorithms.sampling.methods import reservoir, reservoir_weighted

class JobSampler(object):
    def __init__(self, job_posting, lookup, weights=None, show_major_group=True, random_state=None):
        self.job_posting = job_posting
        self.lookup = lookup
        self.weights = weights
        self.show_major_group = show_major_group
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)

    def _transform_iterator(self, job_iter, lookup, show_major_group=True):
        for job in job_iter:
            if show_major_group:
                yield (job[0], lookup[job[1][0]][:2])
            else:
                yield (job[0], lookup[job[1][0]])

    def sample(self, k):
        it = self._transform_iterator(self.job_posting, self.lookup, self.show_major_group)
        if self.weights:
            return list(reservoir_weighted(it, k, self.weights))
        else:
            return list(reservoir(it, k))
