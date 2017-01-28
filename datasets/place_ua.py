"""Retrieve Census Place->Urbanized Area crosswalk"""
from collections import defaultdict
import unicodecsv as csv
import logging
import requests
import us
import re

from utils.fs import cache_json

URL = 'http://www2.census.gov/geo/docs/maps-data/data/rel/ua_place_rel_10.txt'
ABBR_LOOKUP = us.states.mapping('fips', 'abbr')
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


@cache_json('place_ua_lookup.json')
def place_ua():
    """
    Construct a Place->UA Lookup table from Census data
    Returns: dict
    { StateCode: { PlaceName: UA Code } }
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
        place_name = row[4]
        place_fips = row[3]

        if place_fips == '99999' or ua == '99999':
            not_designated += 1
            continue

        cleaned_place_name = re.sub(r'\([^)]*\)', '', place_name).rstrip()
        suffix_found = False
        for suffix in SUFFIXES:
            if cleaned_place_name.endswith(suffix):
                cleaned_place_name = cleaned_place_name.replace(suffix, '').rstrip().lower()
                lookup[ABBR_LOOKUP[state_fips]][cleaned_place_name] = ua
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
