from skills_ml.job_postings.common_schema import \
    generate_job_postings_from_s3,\
    generate_job_postings_from_s3_multiple_prefixes,\
    JobPostingCollectionFromS3,\
    JobPostingCollectionSample, \
    batches_generator, \
    BatchGenerator
import json
import moto
import boto
from unittest import mock
import random
import string
import unittest
import collections

raise_error = True


def assert_json_strings_equal(fs, jp):
    json_strings = [json.dumps(obj) for obj in jp]
    assert set(json_strings) == set(fs)


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
            key.set_contents_from_string(json.dumps({'test': 'test'}))

        postings = [posting for posting in generate_job_postings_from_s3(
            s3_conn,
            f'{bucket_name}/{path}/{quarter}'
        )]
        self.assertEqual(postings, [{'test': 'test'}] * 2)

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
            key.set_contents_from_string(json.dumps({'test': 'test'}))

        with mock.patch('boto.s3.key.Key.get_contents_to_file') as patched:
            def maybe_give_jobposting(fh, cb=None):
                global raise_error
                raise_error = not raise_error
                if raise_error:
                    raise ConnectionResetError('Connection reset by peer')
                else:
                    fh.write(json.dumps({'test': 'test'}).encode('utf-8'))

            patched.side_effect = maybe_give_jobposting
            postings = [posting for posting in generate_job_postings_from_s3(
                s3_conn,
                f'{bucket_name}/{path}/{quarter}'
            )]
            self.assertEqual(postings, [{'test': 'test'}] * 2)

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

        file_contents = []
        for f in files:
            key = boto.s3.key.Key(
                bucket=bucket,
                name='{}/{}/{}'.format(path, quarter, f)
            )
            contents = json.dumps({'hash': f})
            key.set_contents_from_string(contents)
            file_contents.append(contents)

        jp = generate_job_postings_from_s3(s3_conn, f'{bucket_name}/{path}/{quarter}')
        assert_json_strings_equal(file_contents, jp)

        jp = generate_job_postings_from_s3(s3_conn, f'{bucket_name}/{path}/{quarter}/VA_')
        assert_json_strings_equal([j for j in file_contents if 'VA_' in j], jp)

        jp = generate_job_postings_from_s3(s3_conn, f'{bucket_name}/{path}/{quarter}/NLX_')
        assert_json_strings_equal([j for j in file_contents if 'NLX_' in j], jp)

        jp = generate_job_postings_from_s3(s3_conn, f'{bucket_name}/{path}/{quarter}/CB_')
        assert_json_strings_equal([j for j in file_contents if 'CB_' in j], jp)

        jp = generate_job_postings_from_s3_multiple_prefixes(s3_conn, [f'{bucket_name}/{path}/{quarter}/NLX_', f'{bucket_name}/{path}/{quarter}/CB_'])
        assert_json_strings_equal([j for j in file_contents if ('NLX_' in j or 'CB_' in j)], jp)

    @moto.mock_s3_deprecated
    def test_JobPostingCollectionFromS3(self):
        s3_conn = boto.connect_s3()
        bucket_name = 'test-bucket'
        path = 'postings'
        bucket = s3_conn.create_bucket(bucket_name)
        files = []
        for i in range(0, 9):
            random_hash = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
            files.append(random_hash)

        for f in files:
            key = boto.s3.key.Key(
                bucket=bucket,
                name='{}/{}'.format(path, f)
            )
            key.set_contents_from_string(json.dumps({'hash': f}))

        path_string = f'{bucket_name}/{path}'
        jp = JobPostingCollectionFromS3(s3_conn, path_string)
        self.assertEqual(len(list(jp)), len(files))
        self.assertEqual(jp.metadata, {'job postings': { 's3_paths': [path_string]}})


    def test_JobPostingCollectionSample(self):
        job_postings = JobPostingCollectionSample()
        list_of_postings = list(job_postings)
        self.assertEqual(len(list_of_postings), 50)
        for posting in list_of_postings:
            self.assertIsInstance(posting, dict)
            self.assertEqual(posting['@type'], 'JobPosting')
            self.assertIn('title', posting)
            self.assertIn('description', posting)
        self.assertIn('job postings', job_postings.metadata)

    def test_batch_generator(self):
        job_postings = JobPostingCollectionSample()
        batch_iter = next(batches_generator(job_postings, 10))
        # Each bacth produced by batched_generator() is still an iterator
        assert isinstance(batch_iter, collections.Iterator)
        batch_iter = list(batch_iter)
        assert len(list(batch_iter)) == 10

        job_postings = JobPostingCollectionSample()
        batch_tuple = next(BatchGenerator(job_postings, 10))
        # Each batch produced by BatchGenerator() is a tuple
        assert isinstance(batch_tuple, tuple)
        assert len(list(batch_tuple)) == 10

        self.assertListEqual(list(batch_iter), list(batch_tuple))
