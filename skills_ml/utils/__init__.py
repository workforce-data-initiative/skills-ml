"""Common utilities"""
import json
import hashlib
import datetime
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
