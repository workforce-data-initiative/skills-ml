import numpy as np
import heapq as hq

def reservoir(it, k):
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


def reservoir_weighted(it, k):
    weights = {'11': 2.0, '13': 3.5, '29': 1.2}
    heap = []
    hkey = lambda w: -np.random.exponential(1.0/w)
    for i, datum in enumerate(it):
        weight = weights[lookup[datum[1][0]][:2]]
        if len(heap) < k:
            hq.heappush(heap, (hkey(weight), datum))
        elif hkey(weight) > heap[0][0]:
            hq.heapreplace(heap, (hkey(weight), datum))
    while len(heap) > 0:
        yield hq.heappop(heap)[1]

