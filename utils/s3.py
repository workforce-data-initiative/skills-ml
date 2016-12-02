"""
Common S3 utilities
"""
import boto


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
        name='{}/{}'.format(prefix, filename.replace('/', '_'))
    )
    key.set_contents_from_filename(filename)
