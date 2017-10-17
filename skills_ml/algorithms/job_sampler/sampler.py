import numpy as np
import heapq as hq
import random

def reservoir(it, k):
    """
    Reservoir sampling with Random Sort from a job posting iterator.

    Randomly choosing a sample of k items from a streaming iterator. Using random sort to implement the algorithm.
    Basically, it's assigning random number as keys to each item and maintain k items with minimum value for keys,
    which equals to assigning a random number to each item as key and sort items using these keys and take top k items.

    Args:
        it (iterator): Job posting iterator to sample from
        k (int): Sample size

    Returns:
        list: The result sample of k items.
    """
    it = iter(it)
    result = []
    for i, datum in enumerate(it):
        if i < k:
            result.append(datum)
        else:
            j = random.randint(0, i-1)
            if j < k:
                result[j] = datum
    return result


def reservoir_weighted(it, k, weights):
    """
    Weighted reservoir Sampling from job posting iterator.

    Randomly choosing a sample of k items from a streaming iterator based on the weights.

    Args:
        it (iterator): Job posting iterator to sample from. The format should be (job_posting, label)
        k (int): Sample size
        weights (dict): weight
    """
    heap = []
    hkey = lambda w: -np.random.exponential(1.0 / w)
    for i, datum in enumerate(it):
        weight = weights[datum[1]]
        if len(heap) < k:
            hq.heappush(heap, (hkey(weight), datum))
        elif hkey(weight) > heap[0][0]:
            hq.heapreplace(heap, (hkey(weight), datum))
    while len(heap) > 0:
        yield hq.heappop(heap)[1]

class JobSampler(object):
    def __init__(self, job_posting, lookup, weights=None, high_level=True):
        self.job_posting = job_posting
        self.lookup = lookup
        self.weights = weights
        self.high_level = high_level

    def _transform_iterator(self, job_iter, lookup, high_level=True):
        for job in job_iter:
            if high_level:
                yield (job[0], lookup[job[1][0]][:2])
            else:
                yield (job[0], lookup[job[1][0]])

    def sample(self, k):
        it = self._transform_iterator(self.job_posting, self.lookup, self.high_level)
        if self.weights:
            return list(reservoir_weighted(it, k, self.weights))
        else:
            return reservoir(it, k)



