import json

import moto
import boto
from freezegun import freeze_time

from skills_utils.s3 import split_s3_path

from skills_ml.algorithms.aggregators.dataset_transform import DatasetStatsCounter, \
    DatasetStatsAggregator, GlobalStatsAggregator


def add_s3_content(s3_conn, key_data):
    for path, data in key_data.items():
        bucket_name, key_name = split_s3_path(path)
        bucket = s3_conn.create_bucket(bucket_name)
        key = boto.s3.key.Key(
            bucket=bucket,
            name=key_name
        )
        key.set_contents_from_string(data)


def test_dataset_stats_counter():
    counter = DatasetStatsCounter(quarter='2014Q1', dataset_id='VA')
    counter.track(
        input_document={'jobtitle': 'test', 'jobdesc': 'test'},
        output_document={'title': 'test', 'description': 'test'}
    )
    counter.track(
        input_document={'jobtitle': 'test', 'jobdesc': '', 'extra': 'test'},
        output_document={'title': 'test', 'description': ''}
    )
    assert counter.stats['total'] == 2
    assert counter.stats['output_counts']['title'] == 2
    assert counter.stats['output_counts']['description'] == 1
    assert counter.stats['input_counts']['jobtitle'] == 2
    assert counter.stats['input_counts']['jobdesc'] == 1
    assert counter.stats['input_counts']['extra'] == 1

    with moto.mock_s3():
        with freeze_time('2017-01-10'):
            s3_conn = boto.connect_s3()
            s3_conn.create_bucket('test-bucket')
            counter.save(s3_conn, 'test-bucket/stats')

            key = s3_conn.get_bucket('test-bucket')\
                .get_key('stats/quarterly/VA_2014Q1')

        expected_stats = {
            'total': 2,
            'output_counts': {
                'title': 2,
                'description': 1
            },
            'input_counts': {
                'jobtitle': 2,
                'jobdesc': 1,
                'extra': 1
            },
            'output_percentages': {
                'title': 1.0,
                'description': 0.5
            },
            'input_percentages': {
                'jobtitle': 1.0,
                'jobdesc': 0.5,
                'extra': 0.5
            },
            'last_updated': '2017-01-10T00:00:00',
            'quarter': '2014Q1',
        }
        assert json.loads(key.get_contents_as_string().decode('utf-8')) == expected_stats


def test_dataset_stats_counter_empty():
    counter = DatasetStatsCounter(quarter='2013Q1', dataset_id='VA')
    with moto.mock_s3():
        with freeze_time('2017-01-10'):
            s3_conn = boto.connect_s3()
            s3_conn.create_bucket('test-bucket')
            counter.save(s3_conn, 'test-bucket/stats')

            key = s3_conn.get_bucket('test-bucket')\
                .get_key('stats/quarterly/VA_2013Q1')

        expected_stats = {
            'total': 0,
            'output_counts': {},
            'input_counts': {},
            'output_percentages': {},
            'input_percentages': {},
            'last_updated': '2017-01-10T00:00:00',
            'quarter': '2013Q1',
        }
        assert json.loads(key.get_contents_as_string().decode('utf-8')) == expected_stats


def sample_quarter_stats(quarter):
    return {
        'total': 2,
        'output_counts': {
            'title': 2,
            'description': 1
        },
        'input_counts': {
            'jobtitle': 2,
            'jobdesc': 1,
            'extra': 1
        },
        'output_percentages': {
            'title': 1.0,
            'description': 0.5

        },
        'input_percentages': {
            'jobtitle': 1.0,
            'jobdesc': 0.5,
            'extra': 0.5
        },
        'quarter': quarter,
        'last_updated': '2017-01-10T00:00:00',
    }


def sample_dataset_stats():
    return {
        'total': 4,
        'output_counts': {
            'title': 4,
            'description': 2
        },
        'input_counts': {
            'jobtitle': 4,
            'jobdesc': 2,
            'extra': 2
        },
        'output_percentages': {
            'title': 1.0,
            'description': 0.5

        },
        'input_percentages': {
            'jobtitle': 1.0,
            'jobdesc': 0.5,
            'extra': 0.5
        },
        'last_updated': '2017-01-10T00:00:00',
        'quarters': {
            '2014Q1': sample_quarter_stats('2014Q1'),
            '2014Q2': sample_quarter_stats('2014Q2')
        }
    }


def test_dataset_stats_aggregator():
    with moto.mock_s3():
        s3_conn = boto.connect_s3()
        aggregator = DatasetStatsAggregator(dataset_id='CB', s3_conn=s3_conn)

        add_s3_content(
            s3_conn,
            {
                'test-bucket/stats/quarterly/CB_2014Q1':
                    json.dumps(sample_quarter_stats('2014Q1')),
                'test-bucket/stats/quarterly/CB_2014Q2':
                    json.dumps(sample_quarter_stats('2014Q2')),
                'test-bucket/stats/quarterly/VA_2014Q1':
                    json.dumps(sample_quarter_stats('2014Q1')),
            }
        )

        with freeze_time('2017-01-10'):
            aggregator.run('test-bucket/stats')

        expected_stats = sample_dataset_stats()
        key = s3_conn.get_bucket('test-bucket')\
            .get_key('stats/dataset_summaries/CB.json')
        assert json.loads(key.get_contents_as_string().decode('utf-8')) == expected_stats


def test_global_stats_aggregator():
    with moto.mock_s3():
        s3_conn = boto.connect_s3()
        aggregator = GlobalStatsAggregator(s3_conn=s3_conn)

        add_s3_content(
            s3_conn,
            {
                'test-bucket/stats/dataset_summaries/CB.json':
                    json.dumps(sample_dataset_stats()),
                'test-bucket/stats/dataset_summaries/VA.json':
                    json.dumps(sample_dataset_stats()),
            }
        )

        with freeze_time('2017-01-10'):
            aggregator.run('test-bucket/stats')

        expected_stats = {
            'total': 8,
            'output_counts': {
                'title': 8,
                'description': 4
            },
            'output_percentages': {
                'title': 1.0,
                'description': 0.5

            },
            'last_updated': '2017-01-10T00:00:00',
        }
        key = s3_conn.get_bucket('test-bucket').get_key('stats/summary.json')
        assert json.loads(key.get_contents_as_string().decode('utf-8')) == expected_stats
