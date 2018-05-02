from skills_ml.algorithms.geocoders import CachedGeocoder
from skills_ml.algorithms.geocoders.cbsa import CachedCBSAFinder
from skills_ml.storage import S3Store
import json
from unittest.mock import MagicMock, call, patch
from collections import namedtuple
import moto
import boto3


@moto.mock_s3
def test_geocode_cacher():
    with patch('time.sleep') as time_mock:
        with open('tests/sample_geocode_result.json') as f:
            sample_geocode_result = json.load(f)
        client = boto3.resource('s3')
        client.create_bucket(Bucket='geobucket')
        cache_storage = S3Store('geobucket')
        cache_fname = 'cbsas.json'
        geocode_result = namedtuple('GeocodeResult', ['json'])
        geocode_func = MagicMock(
            return_value=geocode_result(json=sample_geocode_result)
        )
        geocoder = CachedGeocoder(
            cache_storage=cache_storage,
            cache_fname=cache_fname,
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

        new_geocoder = CachedGeocoder(
            cache_storage=cache_storage,
            cache_fname=cache_fname,
            geocode_func=geocode_func,
            sleep_time=1
        )
        assert new_geocoder.all_cached_geocodes == {
            'Canarsie, NY': sample_geocode_result,
            'Poughkeepsie, NY': sample_geocode_result,
        }


@moto.mock_s3
def test_geocode_search_strings():
    with open('tests/sample_geocode_result.json') as f:
        sample_geocode_result = json.load(f)
    client = boto3.resource('s3')
    client.create_bucket(Bucket='geobucket')
    cache_storage = S3Store('geobucket')
    cache_fname = 'cbsas.json'
    geocode_result = namedtuple('GeocodeResult', ['json'])
    geocode_func = MagicMock(
        return_value=geocode_result(json=sample_geocode_result)
    )
    geocoder = CachedGeocoder(
        cache_storage=cache_storage,
        cache_fname=cache_fname,
        geocode_func=geocode_func,
        sleep_time=0
    )
    geocoder.geocode_search_strings_and_save(['string1', 'string2'])

    new_geocoder = CachedGeocoder(
        cache_storage=cache_storage,
        cache_fname=cache_fname,
    )
    assert next(iter(new_geocoder.all_cached_geocodes.values()))\
        == sample_geocode_result


@moto.mock_s3
def test_cbsa_finder_onehit():
    client = boto3.resource('s3')
    client.create_bucket(Bucket='geobucket')
    shapefile_name = 'tests/sample_cbsa_shapefile.shp'
    cache_storage = S3Store('geobucket')
    cache_fname = 'cbsas.json'
    finder = CachedCBSAFinder(
        cache_storage=cache_storage,
        cache_fname=cache_fname,
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
    client = boto3.resource('s3')
    client.create_bucket(Bucket='geobucket')
    shapefile_name = 'tests/sample_cbsa_shapefile.shp'
    cache_storage = S3Store('geobucket')
    cache_fname = 'cbsas.json'
    finder = CachedCBSAFinder(
        cache_storage=cache_storage,
        cache_fname=cache_fname,
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
    client = boto3.resource('s3')
    client.create_bucket(Bucket='geobucket')
    shapefile_name = 'tests/sample_cbsa_shapefile.shp'
    cache_storage = S3Store('geobucket')
    cache_fname = 'cbsas.json'
    finder = CachedCBSAFinder(
        cache_storage=cache_storage,
        cache_fname=cache_fname,
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
    client = boto3.resource('s3')
    client.create_bucket(Bucket='geobucket')
    cache_storage = S3Store('geobucket')
    cache_fname = 'cbsas.json'
    cbsa_finder = CachedCBSAFinder(
        cache_storage=cache_storage,
        cache_fname=cache_fname,
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

    new_finder = CachedCBSAFinder(
        cache_storage=cache_storage,
        cache_fname=cache_fname,
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
def test_cbsa_finder_empty_cache():
    client = boto3.resource('s3')
    geobucket = client.create_bucket(Bucket='geobucket')
    cache_storage = S3Store('geobucket')
    cache_fname = 'cbsas.json'
    cbsa_finder = CachedCBSAFinder(
        cache_storage=cache_storage,
        cache_fname=cache_fname,
        shapefile_name='tests/sample_cbsa_shapefile.shp'
    )
    # set the cache to something that JSON loads as None, not empty dict
    geobucket.put_object(Body='', Key='cbsas.json')
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

    new_finder = CachedCBSAFinder(
        cache_storage=cache_storage,
        cache_fname=cache_fname,
        shapefile_name='tests/sample_cbsa_shapefile.shp'
    )
    print(new_finder.all_cached_cbsa_results._storage)
    assert new_finder.all_cached_cbsa_results == {
        'East of Charlotte, NC': [
            '16740',
            'Charlotte-Concord-Gastonia, NC-SC Metro Area',
        ],
        'Flushing, NY': None
    }
