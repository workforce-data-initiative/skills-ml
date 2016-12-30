import logging
import tempfile

from config import config
from utils.s3 import split_s3_path
from utils.s3 import log_download_progress


def job_postings(s3_conn, quarter, s3_path=None):
    """
    Stream all job listings from s3 for a given quarter
    Args:
        s3_conn: a boto s3 connection
        quarter: a string representing a quarter (2015Q1)
        s3_path (optional): path to the job listings. Defaults to config file

    Yields:
        string in json format representing the next job listing
            Refer to sample_job_listing.json for example structure
    """
    s3_path = s3_path or config['job_postings']['s3_path']
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
