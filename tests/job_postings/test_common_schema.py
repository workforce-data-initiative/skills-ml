from skills_ml.job_postings.common_schema import job_postings, job_postings_chain, JobPostingGenerator
import moto
import boto
from unittest import mock
import random
import string
import unittest

raise_error = True


class CommonSchemaTestCase(unittest.TestCase):
    @moto.mock_s3_deprecated
    def test_job_postings(self):
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
        for func in [job_postings]:
            postings = [posting for posting in func(
                s3_conn,
                quarter,
                '{}/{}'.format(bucket_name, path)
            )]
            self.assertEqual(postings, ['test'] * 2)



    @moto.mock_s3_deprecated
    def test_job_postings_retry(self):
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
            self.assertEqual(postings, ['test'] * 2)


    @moto.mock_s3_deprecated
    def test_job_postings_choosing_source(self):
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
        self.assertEqual(set(jp), set(files))

        jp = job_postings(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'va')
        self.assertEqual(set([j for j in files if 'VA' in j.split('_')]), set(jp))

        jp = job_postings(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'nlx')
        self.assertEqual(set([j for j in files if 'NLX' in j.split('_')]), set(jp))

        jp = job_postings(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'cb')
        self.assertEqual(set([j for j in files if 'CB' in j.split('_')]), set(jp))

        self.assertRaises(ValueError, lambda: set(job_postings(s3_conn, quarter, '{}/{}'.format(bucket_name, path), 'abc')))

        jp = job_postings(s3_conn, quarter, '{}/{}'.format(bucket_name, path), ['nlx', 'cb'])
        self.assertEqual(set([j for j in files if ('NLX' in j.split('_') or 'CB' in j.split('_'))]), set(jp))


    @moto.mock_s3_deprecated
    def test_job_postings_chain(self):
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

        jp = job_postings_chain(s3_conn, quarters, '{}/{}'.format(bucket_name, path), 'all')
        self.assertEqual(len(list(jp)), len(files))

        jp = job_postings_chain(s3_conn, quarters, '{}/{}'.format(bucket_name, path), 'nlx')
        self.assertEqual(len(list(jp)), len([j for j in files if 'NLX' in j.split('_')]))

        jp = job_postings_chain(s3_conn, quarters, '{}/{}'.format(bucket_name, path), ['va', 'nlx'])
        self.assertEqual(len(list(jp)), len([j for j in files if ('VA' in j.split('_') or 'NLX' in j.split('_'))]))


    @moto.mock_s3_deprecated
    def test_job_postings_generator(self):
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

        jp = JobPostingGenerator(s3_conn, quarters, '{}/{}'.format(bucket_name, path), 'all')
        self.assertEqual(len(list(jp)), len(files))

        jp = JobPostingGenerator(s3_conn, quarters, '{}/{}'.format(bucket_name, path), 'nlx')
        self.assertEqual(len(list(jp)), len([j for j in files if 'NLX' in j.split('_')]))

        jp = JobPostingGenerator(s3_conn, quarters, '{}/{}'.format(bucket_name, path), ['va', 'nlx'])
        self.assertEqual(len(list(jp)), len([j for j in files if ('VA' in j.split('_') or 'NLX' in j.split('_'))]))

        self.assertEqual(jp.quarters, quarters)
