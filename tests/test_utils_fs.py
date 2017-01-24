from utils.fs import cache_json, CACHE_DIRECTORY
import os
import json


def test_cache_json():
    counter = {'times_called': 0}

    @cache_json('test.txt')
    def a_function(counter):
        counter['times_called'] += 1
        return '6'

    fname = CACHE_DIRECTORY + 'test.txt'
    if os.path.exists(fname):
        os.unlink(fname)
    assert a_function(counter) == '6'
    with open(fname) as infile:
        assert json.load(infile) == '6'
    # run the function again and make sure the cached version is used
    assert a_function(counter) == '6'
    assert counter['times_called'] == 1
