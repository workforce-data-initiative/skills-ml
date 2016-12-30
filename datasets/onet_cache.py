import contextlib
import os

import boto

from config import config
from utils.s3 import split_s3_path


class OnetCache(object):
    """
    An object that downloads and caches ONET files from S3
    """
    def __init__(self, s3_conn, s3_path=None, cache_dir=None):
        """
        Args:
            s3_conn: a boto s3 connection
            s3_path (optional): path to the onet directory, defaults to config
            cache_dir (optional): directory to cache files, defaults to config
        """
        self.s3_conn = s3_conn
        self.cache_dir = cache_dir or config['onet']['cache_dir']
        self.s3_path = s3_path or config['onet']['s3_path']
        self.bucket_name, self.prefix = split_s3_path(self.s3_path)

    @contextlib.contextmanager
    def ensure_file(self, filename):
        """
        Ensures that the given ONET data file is present, either by
        using a cached copy or downloading from S3

        Args:
            filename: unpathed filename of an ONET file (Skills.txt)

        Yields:
            Full path to file on local filesystem
        """
        full_path = os.path.join(self.cache_dir, filename)
        if os.path.isfile(full_path):
            yield full_path
        else:
            if not os.path.isdir(self.cache_dir):
                os.mkdir(self.cache_dir)
            bucket = self.s3_conn.get_bucket(self.bucket_name)
            key = boto.s3.key.Key(
                bucket=bucket,
                name='{}/{}'.format(self.prefix, filename)
            )
            key.get_contents_to_filename(full_path)
            yield full_path
