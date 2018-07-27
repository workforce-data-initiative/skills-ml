import contextlib
import os
import csv
import io

import boto

from skills_utils.s3 import split_s3_path
from skills_ml.datasets.onet_source import OnetToMemoryDownloader
from skills_ml.storage import InMemoryStore


class OnetCache(object):
    """
    An object that downloads and caches ONET files from S3
    """
    def __init__(self, s3_conn, s3_path, cache_dir):
        """
        Args:
            s3_conn: a boto s3 connection
            s3_path: path to the onet directory
            cache_dir: directory to cache files
        """
        self.s3_conn = s3_conn
        self.cache_dir = cache_dir
        self.s3_path = s3_path
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


class OnetSiteCache(object):
    """
    An object that downloads files from the ONET site
    """
    def __init__(self, storage=None):
        """
        Args:
            storage: Storage object to cache files
        """
        self.storage = storage or InMemoryStore('')
        self.downloader = OnetToMemoryDownloader()

    def reader(self, filename):
        """
        Ensures that the given ONET data file is present, either by
        using a cached copy or downloading from S3

        Args:
            filename: unpathed filename of an ONET file (Skills)

        Returns:
            csv.DictReader
        """
        if not self.storage.exists(filename):
            self.storage.write(self.downloader.download(filename).encode('utf-8'), filename)
        contents = self.storage.load(filename)
        return csv.DictReader(io.StringIO(contents.decode('utf-8')), delimiter='\t')
