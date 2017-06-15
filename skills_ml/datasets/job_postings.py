import logging
import tempfile
from io import BytesIO

from skills_utils.s3 import split_s3_path
from skills_utils.s3 import log_download_progress


def job_postings(s3_conn, quarter, s3_path):
    """
    Stream all job listings from s3 for a given quarter
    Args:
        s3_conn: a boto s3 connection
        quarter: a string representing a quarter (2015Q1)
        s3_path: path to the job listings.

    Yields:
        string in json format representing the next job listing
            Refer to sample_job_listing.json for example structure
    """
    bucket_name, prefix = split_s3_path(s3_path)
    bucket = s3_conn.get_bucket(bucket_name)
    keys = bucket.list(prefix='{}/{}'.format(prefix, quarter))
    for key in keys:
        logging.info('Extracting job postings from key {}'.format(key.name))
        with tempfile.NamedTemporaryFile() as outfile:
            key.get_contents_to_file(outfile, cb=log_download_progress)
            outfile.seek(0)
            for line in outfile:
                yield line.decode('utf-8')


def job_postings_highmem(s3_conn, quarter, s3_path):
    """
    Stream all job listings from s3 for a given quarter
    Args:
        s3_conn: a boto s3 connection
        quarter: a string representing a quarter (2015Q1)
        s3_path: path to the job listings.

    Yields:
        string in json format representing the next job listing
            Refer to sample_job_listing.json for example structure
    """
    bucket_name, prefix = split_s3_path(s3_path)
    bucket = s3_conn.get_bucket(bucket_name)
    keys = bucket.list(prefix='{}/{}'.format(prefix, quarter))
    for key in keys:
        logging.info('Extracting job postings from key {}'.format(key.name))
        content = key.get_contents_as_string(cb=log_download_progress)
        outfile = BytesIO(content)
        for line in outfile:
            yield line.decode('utf-8')
