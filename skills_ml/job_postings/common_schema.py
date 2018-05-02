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

from typing import Dict, Text, Any, Generator, List, Callable

JobPostingType = Dict[Text, Any]
JobPostingGeneratorType = Generator[JobPostingType, None, None]
MetadataType = Dict[Text, Dict[Text, Any]]


class JobPostingCollectionFromS3(object):
    """
    Stream job posting from s3 for given quarters
    Example:
    ```
    import json
    from airflow.hooks import S3Hook
    s3_conn = S3Hook().get_conn()
    quarters = ['2011Q1', '2011Q2', '2011Q3']
    job_postings_generator = JobPostingCollectionFromS3(s3_conn, quarters, s3_path='open-skills-private/job_postings_common', source="all")
    for job_posting in job_postings_generator:
        print json.loads(job_posting)['title']
    ```

    Attributes:
        s3_conn: a boto s3 connection
        quarters: a list of quarters
        s3_path: path to the job listings
        source: should be a string or a subset of "nlx", "va", "cb" or "all"
    """
    def __init__(self, s3_conn, quarters, s3_path, source='all'):
        self.s3_conn = s3_conn
        self.quarters = quarters
        self.s3_path = s3_path
        self.source = source

    def __iter__(self) -> JobPostingGeneratorType:
        yield from generate_job_postings_from_s3_for_quarters(
            self.s3_conn,
            self.quarters,
            self.s3_path,
            self.source
        )

    @property
    def metadata(self) -> MetadataType:
        return {'job postings':  {'quarters': self.quarters, 'source': self.source}}


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


def generate_job_postings_from_s3_for_quarter(
        s3_conn,
        quarter: Text,
        s3_path: Text,
        source: Text="all"
) -> JobPostingGeneratorType:
    """
    Stream all job listings from s3 for a given quarter
    Args:
        s3_conn: a boto s3 connection
        quarter: a string representing a quarter (2015Q1)
        s3_path: path to the job listings.
        source: should be a string or a subset of "nlx", "va", "cb" or "all"

    Yields:
        string in json format representing the next job listing
            Refer to sample_job_listing.json for example structure
    """
    if isinstance(source, str):
        s = [source]
    else:
        s = source

    if not set(s) <= set(['nlx', 'cb', 'va', 'all']):
        raise ValueError('"{}" is an invalid source!'.format(s))

    retrier = Retrying(
        retry_on_exception=retry_if_io_error,
        wait_exponential_multiplier=100,
        wait_exponential_max=100000
    )
    bucket_name, prefix = split_s3_path(s3_path)
    bucket = s3_conn.get_bucket(bucket_name)
    if isinstance(source, str):
        if source.lower() == "all":
            keys = bucket.list(prefix='{}/{}'.format(prefix, quarter))
        else:
            keys = bucket.list(prefix='{}/{}/{}_'.format(prefix, quarter, source.upper()))
    elif isinstance(source, list):
        keys = []
        for s in source:
            keys.append(bucket.list(prefix='{}/{}/{}_'.format(prefix, quarter, s.upper())))
        keys = chain(*keys)

    for key in keys:
        logging.info('Extracting job postings from key {}'.format(key.name))
        with BytesIO() as outfile:
            retrier.call(key.get_contents_to_file, outfile, cb=log_download_progress)
            outfile.seek(0)
            for line in outfile:
                yield json.loads(line.decode('utf-8'))


def generate_job_postings_from_s3_for_quarters(
        s3_conn,
        quarters: List[Text],
        s3_path: Text,
        source: Text='all'
) -> JobPostingGeneratorType:
    """
    Chain the generators of a list of multiple quarters
    Args:
        s3_conn: a boto s3 connection
        quarters: a list of quarters
        s3_path: path to the job listings
        source: should be a string or a subset of "nlx", "va", "cb" or "all"

    Return:
        a generator that all generators are chained together into
    """
    if not isinstance(quarters, list):
        raise TypeError('quarters should be a list of string, e.g. ["2011Q1", "2011Q2"]')

    for quarter in quarters:
        yield from generate_job_postings_from_s3_for_quarter(s3_conn, quarter, s3_path, source)


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
