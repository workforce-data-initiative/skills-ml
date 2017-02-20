from collections import defaultdict
from utils.fs import cache_json
import csv
import logging
import requests

URL = 'https://s3-us-west-2.amazonaws.com/skills-public/tables/state_table.csv'


@cache_json('negative_dict_lookup.json')
def negative_dict():
    """
    Construct a dictionary of terms that are considered not to be in job title, including
    states, states abv, cities
    Returns: dictionary
    """
    logging.info("Beginning negative dictionary lookup")
    states = []
    download = requests.get(URL)
    reader = csv.reader(download.content.decode('utf-8').splitlines(), delimiter=',')
    next(reader)
    for row in reader:
        states.append(row[1].lower())
        states.append(row[2].lower())

    return {'states': states}
