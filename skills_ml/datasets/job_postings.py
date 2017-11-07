import logging
import tempfile
from retrying import Retrying
from io import BytesIO
from itertools import chain

from skills_utils.s3 import split_s3_path
from skills_utils.s3 import log_download_progress


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
        with tempfile.NamedTemporaryFile() as outfile:
            retrier.call(key.get_contents_to_file, outfile, cb=log_download_progress)
            outfile.seek(0)
            for line in outfile:
                yield line.decode('utf-8')


def job_postings_highmem(s3_conn, quarter, s3_path, source="all"):
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
        with tempfile.NamedTemporaryFile() as outfile:
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
    generators = []
    for quarter in quarters:
        generators.append(job_postings(s3_conn, quarter, s3_path, source))

    job_postings_generator = chain(*generators)

    return job_postings_generator
