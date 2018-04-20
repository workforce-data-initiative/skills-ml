import logging
import tempfile
from retrying import Retrying
from io import BytesIO
from itertools import chain, islice, groupby, count

from skills_utils.s3 import split_s3_path
from skills_utils.s3 import log_download_progress

class JobPostingGenerator(object):
    """
    Stream job posting from s3 for given quarters
    Example:
    ```
    from airflow.hooks import S3Hook
    from skills_ml.job_postings.common_schema import JobPostingGenerator
    s3_conn = S3Hook().get_conn()
    quarters = ['2011Q1', '2011Q2', '2011Q3']
    job_postings_generator = JobPostingGenerator(s3_conn, quarters, s3_path='open-skills-private/job_postings_common', source="all")
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

    def __iter__(self):
        for job_post in job_postings_chain(self.s3_conn, self.quarters, self.s3_path, self.source):
            yield job_post

    @property
    def metadata(self):
        return {'job_postings_generator':  {'quarters': self.quarters, 'source': self.source}}

def retry_if_io_error(exception):
    return isinstance(exception, IOError)


def job_postings(s3_conn, quarter, s3_path, source="all"):
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
    # keys = bucket.list(prefix='{}/{}'.format(prefix, quarter))
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
                yield line.decode('utf-8')


def job_postings_chain(s3_conn, quarters, s3_path, source='all'):
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

    generators = []
    for quarter in quarters:
        generators.append(job_postings(s3_conn, quarter, s3_path, source))

    job_postings_generator = chain(*generators)

    return job_postings_generator


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
