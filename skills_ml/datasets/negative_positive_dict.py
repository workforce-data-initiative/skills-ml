from collections import defaultdict
from skills_utils.fs import cache_json
import unicodecsv as csv
import logging
import requests
import re
import us

PLACEURL = 'http://www2.census.gov/geo/docs/maps-data/data/rel/ua_place_rel_10.txt'
ONETURL = 'https://s3-us-west-2.amazonaws.com/skills-public/pipeline/tables/job_titles_master_table.tsv'
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

@cache_json('negative_positive_dict_lookup.json')
def negative_positive_dict():
    """
    Construct a dictionary of terms that are considered not to be in job title, including
    states, states abv, cities
    Returns: dictionary of set
    """
    logging.info("Beginning negative dictionary build")
    states = []
    states.extend(list(map(lambda x: x.lower(), list(us.states.mapping('name', 'abbr').keys()))))
    states.extend(list(map(lambda x: x.lower(), list(us.states.mapping('name', 'abbr').values()))))

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

    onetjobs = []
    download = requests.get(ONETURL)
    reader = csv.reader(download.content.splitlines(), delimiter='\t')
    next(reader)
    for row in reader:
        onetjobs.append(row[2].lower())
        onetjobs.append(row[3].lower())
    onetjobs = list(set(onetjobs))

    return {'states': states, 'places': places, 'onetjobs': onetjobs}
