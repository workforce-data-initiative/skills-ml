"""
Common S3 utilities
"""
import boto
import logging


def split_s3_path(path):
    """
    Args:
        path: a string representing an s3 path including a bucket
            (bucket_name/prefix/prefix2)
    Returns:
        A tuple containing the bucket name and full prefix)
    """
    return path.split('/', 1)


def upload(s3_conn, filename, s3_path):
    """
    Uploads the given file to s3
    Args:
        s3_conn: a boto s3 connection
        filename: local filename
        s3_path: the destination path on s3
    """
    bucket_name, prefix = split_s3_path(s3_path)
    bucket = s3_conn.get_bucket(bucket_name)

    key = boto.s3.key.Key(
        bucket=bucket,
        name='{}/{}'.format(prefix, filename)
    )
    logging.info('uploading from %s to %s', filename, key)
    key.set_contents_from_filename(filename)


def download(s3_conn, out_filename, s3_path):
    bucket_name, prefix = split_s3_path(s3_path)
    bucket = s3_conn.get_bucket(bucket_name)
    key = boto.s3.key.Key(
        bucket=bucket,
        name=prefix
    )
    logging.info('loading from %s into %s', key, out_filename)
    key.get_contents_to_filename(out_filename, cb=log_download_progress)


def log_download_progress(num_bytes, obj_size):
    logging.info('%s bytes transferred out of %s total', num_bytes, obj_size)
