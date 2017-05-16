"#""Retrieve Urbanized Area->CBSA crosswalk"""
from collections import defaultdict
import unicodecsv as csv
import logging
import requests

from skills_utils.fs import cache_json

URL = 'http://www2.census.gov/geo/docs/maps-data/data/rel/ua_cbsa_rel_10.txt'


@cache_json('ua_cbsa_lookup.json')
def ua_cbsa():
    """
    Construct a UA->CBSA Lookup table from Census data
    Returns: dict
    { UA Fips: [(CBSA FIPS, CBSA Name)] }
    """
    logging.info("Beginning CBSA lookup")
    lookup = defaultdict(list)
    download = requests.get(URL)
    reader = csv.reader(
        download.content.decode('latin-1').encode('utf-8').splitlines(),
        delimiter=','
    )
    not_designated = 0
    total = 0
    # skip header line
    next(reader)
    for row in reader:
        total += 1
        ua_fips = row[0]
        cbsa_fips = row[2]
        cbsa_name = row[3]

        if cbsa_fips == '99999' or ua_fips == '99999':
            not_designated += 1
            continue

        lookup[ua_fips].append((cbsa_fips, cbsa_name))

        logging.info(
            'Done extracting CBSAs %s total rows, %s not designated, %s found',
            total,
            not_designated,
            total - not_designated
        )

    return lookup
