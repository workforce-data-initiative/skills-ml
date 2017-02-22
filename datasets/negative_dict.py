from collections import defaultdict
from utils.fs import cache_json
import unicodecsv as csv
import logging
import requests
import re

STATEURL = 'https://s3-us-west-2.amazonaws.com/skills-public/tables/state_table.csv'
PLACEURL = 'http://www2.census.gov/geo/docs/maps-data/data/rel/ua_place_rel_10.txt'
SUFFIXES = [
    'city',
    'town',
    'village',
    'CDP',
    'zona urbana',
    'comunidad',
    'borough',
    'consolidated government',
    'municipality',
    'unified government',
    'metro government',
    'metropolitan government',
    'urban county',
]
DELIMITERS = ['/', '-', ' City']


@cache_json('negative_dict_lookup.json')
def negative_dict():
    """
    Construct a dictionary of terms that are considered not to be in job title, including
    states, states abv, cities
    Returns: dictionary
    """
    logging.info("Beginning negative dictionary lookup")
    states = []
    download = requests.get(STATEURL)
    reader = csv.reader(download.content.splitlines(), delimiter=',')
    next(reader)
    for row in reader:
        states.append(row[1].lower())
        states.append(row[2].lower())

    places = []
    download = requests.get(PLACEURL)
    reader = csv.reader(download.content.decode('latin-1').encode('utf-8').splitlines(), delimiter=',')
    next(reader)
    for row in reader:
        cleaned_placename = re.sub(r'\([^)]*\)', '', row[4]).rstrip()
        for suffix in SUFFIXES:
            if cleaned_placename.endswith(suffix):
                cleaned_placename = cleaned_placename.replace(suffix, '').rstrip()
        places.append(cleaned_placename.lower())

    places = list(set(places))
    places.remove('not in a census designated place or incorporated place')

    return {'states': set(states), 'places': set(places)}
