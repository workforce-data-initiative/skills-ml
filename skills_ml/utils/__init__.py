"""Common utilities"""
import json
import hashlib
import datetime
import random
import numpy as np

def filename_friendly_hash(inputs):
    def dt_handler(x):
        try:
            if isinstance(x, datetime.datetime) or isinstance(x, datetime.date):
                return x.isoformat()
            if isinstance(x, np.ndarray):
                return str(x)
        except:
            return 0

    return hashlib.md5(
        json.dumps(inputs, default=dt_handler, sort_keys=True)
            .encode('utf-8')
    ).hexdigest()


def itershuffle(iterable, bufsize=1000):
    """Shuffle an iterator with unknown length in a memory efficient way but with less 
    randomizing power. This works by holding `bufsize` items back and yielding them 
    sometime later. It's not theoretically randomized.

    The limitation is that no item can be displaced from its original location by more 
    than the buffer size. If we have a buffer size of 200 and we have 1000 elements, we
    will never randomly pick the 999th element first. Therefore having a big buffer would help.
    """
    iterable = iter(iterable)
    buf = []
    try:
        while True:
            for i in range(random.randint(1, bufsize-len(buf))):
                buf.append(next(iterable))
            random.shuffle(buf)
            for i in range(random.randint(1, bufsize)):
                if buf:
                    yield buf.pop()
                else:
                    break
    except StopIteration:
        random.shuffle(buf)
        while buf:
            yield buf.pop()
        raise StopIteration
