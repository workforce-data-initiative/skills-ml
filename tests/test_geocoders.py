from skills_ml.algorithms.geocoders import S3CachedGeocoder,\
    job_posting_search_string
from skills_ml.algorithms.geocoders.cbsa import S3CachedCBSAFinder
import json
from unittest.mock import MagicMock, call, patch
from collections import namedtuple
import moto
import boto


@moto.mock_s3
def test_geocode_cacher():
    with patch('time.sleep') as time_mock:
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

    assert job_posting_search_string(sample_job_posting) == 'Salisbury, Pennsylvania'


def test_job_posting_weird_region():
    fake_job = {'jobLocation': {'address': {
        'addressLocality': 'Any City',
        'addressRegion': 'Northeastern USA'
    }}}

    assert job_posting_search_string(json.dumps(fake_job)) ==\
        'Any City, Northeastern USA'


def test_job_posting_search_string_only_city():
    fake_job = {'jobLocation': {'address': {'addressLocality': 'City'}}}
    assert job_posting_search_string(json.dumps(fake_job)) == 'City'


@moto.mock_s3
def test_cbsa_finder_onehit():
    s3_conn = boto.connect_s3()
    s3_conn.create_bucket('geobucket')
    shapefile_name = 'tests/sample_cbsa_shapefile.shp'
    finder = S3CachedCBSAFinder(
        s3_conn=s3_conn,
        cache_s3_path='geobucket/cbsas.json',
        shapefile_name=shapefile_name
    )
    sample_input = {
        "lng": -80.8462211,
        "ok": True,
        "location": "East of Charlotte, NC",
        "provider": "osm",
        "country": "United States of America",
        "bbox": {
            "northeast": [35.2268961, -80.8461711],
            "southwest": [35.2267961, -80.8462711]
        },
        "importance": 0.325,
        "quality": "postcode",
        "accuracy": 0.325,
        "address": "NC 28202, United States of America",
        "confidence": 10, "lat": 35.2268461,
        "type": "postcode",
        "place_rank": "25",
        "status_code": 200,
        "status": "OK",
        "place_id": "210190423",
        "encoding": "utf-8",
        "postal": "NC 28202"
    }
    assert finder.query(sample_input) == (
        '16740',
        'Charlotte-Concord-Gastonia, NC-SC Metro Area',
    )


@moto.mock_s3
def test_cbsa_finder_nohits():
    s3_conn = boto.connect_s3()
    s3_conn.create_bucket('geobucket')
    shapefile_name = 'tests/sample_cbsa_shapefile.shp'
    finder = S3CachedCBSAFinder(
        s3_conn=s3_conn,
        cache_s3_path='geobucket/cbsas.json',
        shapefile_name=shapefile_name
    )
    sample_input = {
        "bbox": {
            "northeast": [65.2, 65.8],
            "southwest": [65.2, 65.8]
        },
    }
    assert finder.query(sample_input) == None


@moto.mock_s3
def test_cbsa_finder_twohits():
    s3_conn = boto.connect_s3()
    s3_conn.create_bucket('geobucket')
    shapefile_name = 'tests/sample_cbsa_shapefile.shp'
    finder = S3CachedCBSAFinder(
        s3_conn=s3_conn,
        cache_s3_path='geobucket/cbsas.json',
        shapefile_name=shapefile_name
    )
    sample_input = {
        "bbox": {
            "northeast": [38.00, -81.05],
            "southwest": [35.13, -88.18]
        },
    }
    assert finder.query(sample_input) == (
        '40080',
        'Richmond-Berea, KY Micro Area',
    )


@moto.mock_s3
def test_cbsa_finder_cache():
    s3_conn = boto.connect_s3()
    s3_conn.create_bucket('geobucket')

    cbsa_finder = S3CachedCBSAFinder(
        s3_conn=s3_conn,
        cache_s3_path='geobucket/cbsas.json',
        shapefile_name='tests/sample_cbsa_shapefile.shp'
    )
    geocode_results = {
        'East of Charlotte, NC': {
            "bbox": {
                "northeast": [35.2268961, -80.8461711],
                "southwest": [35.2267961, -80.8462711]
            },
        },
        'Flushing, NY': {
            "bbox": {
                "northeast": [40.7654801, -73.8173791],
                "southwest": [40.7653801, -73.8174791]
            },
        }
    }

    cbsa_finder.find_all_cbsas_and_save(geocode_results)

    new_finder = S3CachedCBSAFinder(
        s3_conn=s3_conn,
        cache_s3_path='geobucket/cbsas.json',
        shapefile_name='tests/sample_cbsa_shapefile.shp'
    )
    assert new_finder.all_cached_cbsa_results == {
        'East of Charlotte, NC': [
            '16740',
            'Charlotte-Concord-Gastonia, NC-SC Metro Area',
        ],
        'Flushing, NY': None
    }


@moto.mock_s3
def test_cbsa_finder_sanity_check():
    s3_conn = boto.connect_s3()
    s3_conn.create_bucket('geobucket')

    cbsa_finder = S3CachedCBSAFinder(
        s3_conn=s3_conn,
        cache_s3_path='geobucket/cbsas.json',
        shapefile_name='tests/sample_cbsa_shapefile.shp'
    )
    geocode_results = {
        'East of Charlotte, NC': {
            "bbox": {
                "northeast": [35.2268961, -80.8461711],
                "southwest": [35.2267961, -80.8462711]
            },
        },
        'Flushing, NY': {
            "bbox": {
                "northeast": [40.7654801, -73.8173791],
                "southwest": [40.7653801, -73.8174791]
            },
        }
    }

    cbsa_finder.find_all_cbsas_and_save(geocode_results)

    new_finder = S3CachedCBSAFinder(
        s3_conn=s3_conn,
        cache_s3_path='geobucket/cbsas.json',
        shapefile_name='tests/sample_cbsa_shapefile.shp'
    )
    assert new_finder.all_cached_cbsa_results == {
        'East of Charlotte, NC': [
            '16740',
            'Charlotte-Concord-Gastonia, NC-SC Metro Area',
        ],
        'Flushing, NY': None
    }
    new_finder.cache = {}
    # simulate something happening to the cache
    new_finder.save()

    assert new_finder.all_cached_cbsa_results == {
        'East of Charlotte, NC': [
            '16740',
            'Charlotte-Concord-Gastonia, NC-SC Metro Area',
        ],
        'Flushing, NY': None
    }
