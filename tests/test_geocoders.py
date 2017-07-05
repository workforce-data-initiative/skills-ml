from skills_ml.algorithms.geocoders import S3CachedGeocoder,\
    job_posting_search_string
import json
from unittest.mock import MagicMock, call, patch
from collections import namedtuple
import moto
import boto


@moto.mock_s3
@patch('time.sleep')
def test_geocode_cacher(time_mock):
    with open('tests/sample_geocode_result.json') as f:
        sample_geocode_result = json.load(f)
    s3_conn = boto.connect_s3()
    s3_conn.create_bucket('geobucket')

    geocode_result = namedtuple('GeocodeResult', ['json'])
    geocode_func = MagicMock(
        return_value=geocode_result(json=sample_geocode_result)
    )
    geocoder = S3CachedGeocoder(
        s3_conn=s3_conn,
        cache_s3_path='geobucket/geocodes.json',
        geocode_func=geocode_func,
        sleep_time=1
    )
    geocoder.geocode('Canarsie, NY')
    geocoder.geocode('Poughkeepsie, NY')
    geocoder.geocode('Canarsie, NY')
    geocoder.save()
    assert geocode_func.call_count == 2
    assert geocode_func.call_args_list == [
        call('Canarsie, NY'),
        call('Poughkeepsie, NY')
    ]
    assert time_mock.call_count == 2

    new_geocoder = S3CachedGeocoder(
        s3_conn=s3_conn,
        cache_s3_path='geobucket/geocodes.json',
        geocode_func=geocode_func,
        sleep_time=1
    )
    assert new_geocoder.all_cached_geocodes == {
        'Canarsie, NY': sample_geocode_result,
        'Poughkeepsie, NY': sample_geocode_result,
    }


@moto.mock_s3
def test_geocode_job_postings():
    with open('tests/sample_geocode_result.json') as f:
        sample_geocode_result = json.load(f)
    with open('sample_job_listing.json') as f:
        sample_job_posting = f.read()
    s3_conn = boto.connect_s3()
    s3_conn.create_bucket('geobucket')

    geocode_result = namedtuple('GeocodeResult', ['json'])
    geocode_func = MagicMock(
        return_value=geocode_result(json=sample_geocode_result)
    )
    geocoder = S3CachedGeocoder(
        s3_conn=s3_conn,
        cache_s3_path='geobucket/geocodes.json',
        geocode_func=geocode_func,
        sleep_time=0
    )
    geocoder.geocode_job_postings_and_save([sample_job_posting, sample_job_posting], save_every=1)

    new_geocoder = S3CachedGeocoder(
        s3_conn=s3_conn,
        cache_s3_path='geobucket/geocodes.json',
    )
    assert next(iter(new_geocoder.all_cached_geocodes.values()))\
        == sample_geocode_result


def test_job_posting_search_string():
    with open('sample_job_listing.json') as f:
        sample_job_posting = f.read()

    assert job_posting_search_string(sample_job_posting) == 'Salisbury, PA'


def test_job_posting_search_string_only_city():
    fake_job = {'jobLocation': {'address': {'addressLocality': 'City'}}}
    assert job_posting_search_string(json.dumps(fake_job)) == 'City'
