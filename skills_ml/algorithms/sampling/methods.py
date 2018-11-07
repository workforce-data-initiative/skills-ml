"""Generic sampling methods"""
import numpy as np
import heapq as hq
import random

def reservoir(it, k):
    """Reservoir sampling with Random Sort from a job posting iterator

    Randomly choosing a sample of k items from a streaming iterator. Using random sort to implement the algorithm.
    Basically, it's assigning random number as keys to each item and maintain k items with minimum value for keys,
    which equals to assigning a random number to each item as key and sort items using these keys and take top k items.

    Args:
        it (iterator): Job posting iterator to sample from
        k (int): Sample size

    Returns:
        generator: The result sample of k items.
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
    while len(result) > 0:
        yield result.pop()


def reservoir_weighted(it, k, weights, key):
    """Weighted reservoir Sampling from job posting iterator

    Randomly choosing a sample of k items from a streaming iterator based on the weights.


    Args:
        it (iterator): Job posting iterator to sample from. The format should be (job_posting, label)
        k (int): Sample size
        weights (dict): a dictionary that has key-value pairs as label-weighting pairs. It expects every
                        label in the iterator to be present as a key in the weights dictionary For example,
                        weights = {'11': 2, '13', 1}. In this case, the label/key is the occupation major
                        group and the value is the weight you want to sample with.

    Returns:
        generator: The result sample of k items from weighted reservori sampling.

    """
    heap = []
    hkey = lambda w: np.power(np.random.uniform(0.0, 1.0), 1.0 / w)
    for i, datum in enumerate(it):
        weight = weights[key(datum)]
        score = hkey(weight)
        if len(heap) < k:
            hq.heappush(heap, (hkey(weight), datum))
        elif score > heap[0][0]:
            hq.heapreplace(heap, (score, datum))
    while len(heap) > 0:
        yield hq.heappop(heap)[1]
