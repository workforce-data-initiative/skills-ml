"""Retrieve County Subdivision->Urbanized Area crosswalk"""
from collections import defaultdict
import unicodecsv as csv
import logging
import requests
import us
import re

from skills_utils.fs import cache_json

URL = 'http://www2.census.gov/geo/docs/maps-data/data/rel/ua_cousub_rel_10.txt'
ABBR_LOOKUP = us.states.mapping('fips', 'abbr')
SUFFIXES = [
    'city',
    'town',
    'township',
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
DELIMITERS = ['-']


@cache_json('cousub_ua_lookup.json')
def cousub_ua(city_cleaner):
    """
    Construct a County Subdivision->UA Lookup table from Census data
    Returns: dict
    { StateCode: { CountySubdivisionName: UA Code } }
    """
    logging.info("Beginning UA lookup")
    lookup = defaultdict(dict)
    download = requests.get(URL)
    reader = csv.reader(download.content.decode('latin-1').encode('utf-8').splitlines(), delimiter=',')
    not_designated = 0
    total = 0
    # skip header line
    next(reader)
    for row in reader:
        total += 1
        state_fips = row[2]
        ua = row[0]
        place_name = row[6]

        if ua == '99999':
            not_designated += 1
            continue

        cleaned_place_name = re.sub(r'\([^)]*\)', '', place_name).rstrip()
        suffix_found = False
        for suffix in SUFFIXES:
            if cleaned_place_name.endswith(suffix):
                cleaned_place_name = cleaned_place_name.replace(suffix, '').rstrip()
                for delimiter in DELIMITERS:
                    if delimiter in cleaned_place_name:
                        places = cleaned_place_name.split(delimiter)
                        for place in places:
                            if place:
                                lookup[ABBR_LOOKUP[state_fips]][city_cleaner(place)] = ua
                        break
                lookup[ABBR_LOOKUP[state_fips]][city_cleaner(cleaned_place_name)] = ua
                suffix_found = True
                break
        if not suffix_found:
            lookup[ABBR_LOOKUP[state_fips]][cleaned_place_name] = ua

    logging.info(
        'Done extracting urbanized areas and urban clusters. %s total rows, %s not designated, %s found',
        total,
        not_designated,
        total - not_designated
    )

    return lookup
