import hashlib


def md5(x):
    return hashlib.md5(x.encode('utf-8')).hexdigest()
