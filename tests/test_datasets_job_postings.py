from skills_ml.datasets.job_postings import job_postings, job_postings_highmem, job_postings_chain
import moto
import boto
from unittest import mock
import random
import string

@moto.mock_s3_deprecated
def test_job_postings():
    s3_conn = boto.connect_s3()
    bucket_name = 'test-bucket'
    path = 'postings'
    quarter = '2014Q1'
    bucket = s3_conn.create_bucket(bucket_name)
    for i in range(0, 2):
        key = boto.s3.key.Key(
            bucket=bucket,
            name='{}/{}/{}'.format(path, quarter, i)
        )
        key.set_contents_from_string('test')

    # both variants of job postings getter should have identical results
    for func in [job_postings, job_postings_highmem]:
        postings = [posting for posting in func(
            s3_conn,
            quarter,
            '{}/{}'.format(bucket_name, path)
        )]
        assert postings == ['test'] * 2

raise_error = True


@moto.mock_s3_deprecated
def test_job_postings_retry():
    s3_conn = boto.connect_s3()
    bucket_name = 'test-bucket'
    path = 'postings'
    quarter = '2014Q1'
    bucket = s3_conn.create_bucket(bucket_name)
    for i in range(0, 2):
        key = boto.s3.key.Key(
            bucket=bucket,
            name='{}/{}/{}'.format(path, quarter, i)
        )
        key.set_contents_from_string('test')

    with mock.patch('boto.s3.key.Key.get_contents_to_file') as patched:
        def maybe_give_jobposting(fh, cb=None):
            global raise_error
            raise_error = not raise_error
            if raise_error:
                raise ConnectionResetError('Connection reset by peer')
            else:
                fh.write(b'test')

        patched.side_effect = maybe_give_jobposting
        postings = [posting for posting in job_postings(
            s3_conn,
            quarter,
            '{}/{}'.format(bucket_name, path)
        )]
        assert postings == ['test'] * 2


@moto.mock_s3_deprecated
def test_job_postings_highmem_retry():
    s3_conn = boto.connect_s3()
    bucket_name = 'test-bucket'
    path = 'postings'
    quarter = '2014Q1'
    bucket = s3_conn.create_bucket(bucket_name)
    for i in range(0, 2):
        key = boto.s3.key.Key(
            bucket=bucket,
            name='{}/{}/{}'.format(path, quarter, i)
        )
        key.set_contents_from_string('test')

    with mock.patch('boto.s3.key.Key.get_contents_as_string') as patched:
        def maybe_give_jobposting(cb=None):
            global raise_error
            raise_error = not raise_error
            if raise_error:
                raise ConnectionResetError('Connection reset by peer')
            else:
                return b'test'

        patched.side_effect = maybe_give_jobposting
        postings = [posting for posting in job_postings_highmem(
            s3_conn,
            quarter,
            '{}/{}'.format(bucket_name, path)
        )]
        assert postings == ['test'] * 2

@moto.mock_s3_deprecated
def test_job_postings_choosing_source():
    s3_conn = boto.connect_s3()
    bucket_name = 'test-bucket'
    path = 'postings'
    quarter = '2014Q1'
    bucket = s3_conn.create_bucket(bucket_name)
    sources = ["CB", "NLX", "VA"]
    files = []
    for i in range(0, 10):
        random_hash = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
        random_source = random.choice(sources)
        files.append(random_source + '_' + random_hash)

    for f in files:
        key = boto.s3.key.Key(
            bucket=bucket,
            name='{}/{}/{}'.format(path, quarter, f)
        )
        key.set_contents_from_string(str(f))

    jp = job_postings(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'all')
    jp = list(jp)
    assert set(jp) == set(files)

    jp = job_postings(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'va')
    assert set([j for j in files if 'VA' in j.split('_')]) == set(jp)

    jp = job_postings(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'nlx')
    assert set([j for j in files if 'NLX' in j.split('_')]) == set(jp)

    jp = job_postings(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'cb')
    assert set([j for j in files if 'CB' in j.split('_')]) == set(jp)

    jp = job_postings(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'abc')
    assert set([j for j in files if 'ABC' in j.split('_')]) == set(jp)

    jp = job_postings(s3_conn, quarter, '{}/{}'.format(bucket_name, path), ['nlx', 'cb'])
    assert set([j for j in files if ('NLX' in j.split('_') or 'CB' in j.split('_'))]) == set(jp)


@moto.mock_s3_deprecated
def test_job_postings_highmem_choosing_source():
    s3_conn = boto.connect_s3()
    bucket_name = 'test-bucket'
    path = 'postings'
    quarter = '2014Q1'
    bucket = s3_conn.create_bucket(bucket_name)
    sources = ["CB", "NLX", "VA"]
    files = []
    for i in range(0, 10):
        random_hash = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
        random_source = random.choice(sources)
        files.append(random_source + '_' + random_hash)

    for f in files:
        key = boto.s3.key.Key(
            bucket=bucket,
            name='{}/{}/{}'.format(path, quarter, f)
        )
        key.set_contents_from_string(str(f))

    jp = job_postings_highmem(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'all')
    assert set(jp) == set(files)

    jp = job_postings_highmem(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'va')
    assert set([j for j in files if 'VA' in j.split('_')]) == set(jp)

    jp = job_postings_highmem(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'nlx')
    assert set([j for j in files if 'NLX' in j.split('_')]) == set(jp)

    jp = job_postings_highmem(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'cb')
    assert set([j for j in files if 'CB' in j.split('_')]) == set(jp)

    jp = job_postings_highmem(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'abc')
    assert set([j for j in files if 'abc' in j.split('_')]) == set(jp)

    jp = job_postings_highmem(s3_conn, quarter, '{}/{}'.format(bucket_name, path), ['va', 'nlx'])
    assert set([j for j in files if ('VA' in j.split('_') or 'NLX' in j.split('_'))]) == set(jp)

@moto.mock_s3_deprecated
def test_job_postings_chain():
    s3_conn = boto.connect_s3()
    bucket_name = 'test-bucket'
    path = 'postings'
    quarters = ['2011Q1', '2011Q2', '2011Q3']
    bucket = s3_conn.create_bucket(bucket_name)
    sources = ["CB", "NLX", "VA"]
    files = []
    for i in range(0, 9):
        random_hash = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
        random_source = random.choice(sources)
        files.append(random_source + '_' + random_hash)

    for i, quarter in enumerate(quarters):
        for f in files[i*3:(i+1)*3]:
            key = boto.s3.key.Key(
                bucket=bucket,
                name='{}/{}/{}'.format(path, quarter, f)
            )
            key.set_contents_from_string(str(f))

    jp = job_postings_chain(s3_conn, quarters, '{}/{}'.format(bucket_name, path), True, 'all')
    assert len(list(jp)) == len(files)

    jp = job_postings_chain(s3_conn, quarters, '{}/{}'.format(bucket_name, path), True, 'nlx')
    assert len(list(jp)) == len([j for j in files if 'NLX' in j.split('_')])

    jp = job_postings_chain(s3_conn, quarters, '{}/{}'.format(bucket_name, path), True, ['va', 'nlx'])
    assert len(list(jp)) == len([j for j in files if ('VA' in j.split('_') or 'NLX' in j.split('_'))])
