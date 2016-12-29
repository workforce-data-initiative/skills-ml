from moto import mock_s3
import boto
import tempfile
from utils.s3 import download, upload
from mock import patch


@mock_s3
def test_download():
    s3_conn = boto.connect_s3()
    bucket_name = 'test-bucket'
    bucket = s3_conn.create_bucket(bucket_name)
    key = boto.s3.key.Key(
        bucket=bucket,
        name='apath/akey'
    )
    key.set_contents_from_string('test')
    s3_path = 'test-bucket/apath/akey'

    with tempfile.NamedTemporaryFile(mode='w+') as f:
        download(s3_conn, f.name, s3_path)
        f.seek(0)
        assert f.read() == 'test'


@patch.dict('utils.s3.config', {'local_mode': False})
@mock_s3
def test_upload():
    s3_conn = boto.connect_s3()
    bucket_name = 'test-bucket'
    bucket = s3_conn.create_bucket(bucket_name)

    with tempfile.NamedTemporaryFile(mode='w+') as f:
        f.write('test')
        f.seek(0)
        s3_path = 'test-bucket/apath/akey'
        upload(s3_conn, f.name, s3_path)
        key = boto.s3.key.Key(
            bucket=bucket,
            name='apath/akey/{}'.format(f.name.replace('/', '_'))
        )
        s = key.get_contents_as_string()
        assert s.decode('utf-8') == 'test'


@patch.dict('utils.s3.config', {'local_mode': True})
@mock_s3
def test_upload_localmode():
    s3_conn = boto.connect_s3()
    bucket_name = 'test-bucket'
    bucket = s3_conn.create_bucket(bucket_name)

    with tempfile.NamedTemporaryFile(mode='w+') as f:
        f.write('test')
        f.seek(0)
        s3_path = 'test-bucket/apath/akey'
        upload(s3_conn, f.name, s3_path)
        key = boto.s3.key.Key(
            bucket=bucket,
            name='apath/akey/{}'.format(f.name.replace('/', '_'))
        )
        assert not key.exists()
