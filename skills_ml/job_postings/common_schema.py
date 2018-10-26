"""A variety of common-schema job posting collections.

Each class in this module should implement a generator that yields job postings (in the common schema, as a JSON string), and has a 'metadata' attribute so any users of the job postings can inspect meaningful metadata about the postings.
"""
import logging
from retrying import Retrying
from io import BytesIO
from itertools import chain, islice

from skills_utils.s3 import split_s3_path
from skills_utils.s3 import log_download_progress
from skills_ml.job_postings.raw.virginia import VirginiaTransformer

import json
import os
import gzip

from typing import Dict, Text, Any, Generator

JobPostingType = Dict[Text, Any]
JobPostingGeneratorType = Generator[JobPostingType, None, None]
MetadataType = Dict[Text, Dict[Text, Any]]


class JobPostingCollectionFromS3(object):
    """
    Stream job posting from s3.

    Expects that each will be stored in JSON format, one job posting per line.
    The s3_path given will be iterated through as a prefix, so job postings may be
    partitioned under that prefix however you choose.
    It will look in every file under that prefix.

    Example:
    ```
    import json
    from airflow.hooks import S3Hook
    from skills_ml.job_postings.common_schema import JobPostingGenerator
    s3_conn = S3Hook().get_conn()
    job_postings_generator = JobPostingCollectionFromS3(s3_conn, s3_path='my-bucket/job_postings_common_schema')
    for job_posting in job_postings_generator:
        print(job_posting['title'])
    ```

    Attributes:
        s3_conn: a boto s3 connection
        s3_path: path to the job listings. there may be multiple
    """
    def __init__(self, s3_conn, s3_paths, extra_metadata=None):
        self.s3_conn = s3_conn
        self.s3_paths = s3_paths
        if not isinstance(self.s3_paths, list):
            self.s3_paths = [self.s3_paths]
        if not extra_metadata:
            self.extra_metadata = {}

    def __iter__(self) -> JobPostingGeneratorType:
        yield from generate_job_postings_from_s3_multiple_prefixes(self.s3_conn, self.s3_paths)

    @property
    def metadata(self) -> MetadataType:
        """Metadata describing the source/s of the job postings"""
        metadata = {
            's3_paths': self.s3_paths,
        }
        metadata.update(self.extra_metadata)

        return {'job postings': metadata }


class JobPostingCollectionSample(object):
    """Stream a finite number of job postings stored within the library.

    Example:
    ```
    import json

    job_postings = JobPostingCollectionSample()
    for job_posting in job_postings:
        print(json.loads(job_posting)['title'])

    Meant to provide a dependency-less example of common schema job postings
    for introduction to the library

    Args:
        num_records (int): The maximum number of records to return. Defaults to 50 (all postings available)
    """
    def __init__(self, num_records:int=50):
        if num_records > 50:
            logging.warning('Cannot provide %s records as a maximum of 50 are available', num_records)
            num_records = 50
        full_filename = os.path.join(os.path.dirname(__file__), '../../50_sample.json.gz')
        f = gzip.GzipFile(filename=full_filename)
        self.lines = f.read().decode('utf-8').split('\n')[0:num_records]
        self.transformer = VirginiaTransformer(partner_id='VA')

    def __iter__(self) -> JobPostingGeneratorType:
        for line in self.lines:
            if line:
                yield self.transformer._transform(json.loads(line))

    @property
    def metadata(self) -> MetadataType:
        return {'job postings': {
            'downloaded_from': 'http://opendata.cs.vt.edu/dataset/openjobs-jobpostings',
            'month': '2016-07',
            'purpose': 'testing'
        }}


def retry_if_io_error(exception):
    return isinstance(exception, IOError)


def generate_job_postings_from_s3(
        s3_conn,
        s3_prefix: Text,
) -> JobPostingGeneratorType:
    """
    Stream all job listings from s3
    Args:
        s3_conn: a boto s3 connection
        s3_prefix: path to the job listings.

    Yields:
        string in json format representing the next job listing
            Refer to sample_job_listing.json for example structure
    """
    retrier = Retrying(
        retry_on_exception=retry_if_io_error,
        wait_exponential_multiplier=100,
        wait_exponential_max=100000
    )
    bucket_name, prefix = split_s3_path(s3_prefix)
    bucket = s3_conn.get_bucket(bucket_name)
    keys = bucket.list(prefix=prefix)

    for key in keys:
        logging.info('Extracting job postings from key {}'.format(key.name))
        with BytesIO() as outfile:
            retrier.call(key.get_contents_to_file, outfile, cb=log_download_progress)
            outfile.seek(0)
            for line in outfile:
                yield json.loads(line.decode('utf-8'))


def generate_job_postings_from_s3_multiple_prefixes(
        s3_conn,
        s3_prefixes: Text,
) -> JobPostingGeneratorType:
    """
    Chain the generators of a list of multiple quarters
    Args:
        s3_conn: a boto s3 connection
        s3_prefixes: paths to job listings

    Return:
        a generator that all generators are chained together into
    """
    if not isinstance(s3_prefixes, list):
        raise TypeError('s3_prefixes should be a list of string, e.g. ["2011Q1", "2011Q2"]')

    for s3_prefix in s3_prefixes:
        yield from generate_job_postings_from_s3(s3_conn, s3_prefix)


def batches_generator(iterable, batch_size):
    """
    Batch generator
    Args:
        iterable: an iterable
        batch_size: batch size
    """
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, batch_size)
        yield chain([next(batchiter)], batchiter)


class BatchGenerator(object):
    def __init__(self, iterable, batch_size):
        self.sourceiter = iterable
        self.batch_size = batch_size
        self.batches_generator = batches_generator(self.sourceiter, self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        return tuple(next(self.batches_generator))


def get_onet_occupation(job_posting):
    """Retrieve the occupation from the job posting

    First checks the custom 'onet_soc_code' key,
    then the standard 'occupationalCategory' key,
    and falls back to the unknown occupation
    """
    return job_posting.get('onet_soc_code', job_posting.get('occupationalCategory', '99-9999.00'))
