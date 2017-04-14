"""Retrieve county->CBSA crosswalk file from the NBER"""
from collections import defaultdict
import unicodecsv as csv
import logging
import requests

from skills_utils.fs import cache_json

URL = 'http://www.nber.org/cbsa-msa-fips-ssa-county-crosswalk/2016/cbsatocountycrosswalk2016.csv'


@cache_json('cbsa_lookup.json')
def cbsa_lookup():
    """
    Construct a County->CBSA Lookup table from NBER data
    Returns: dict
        each key is a (State Code, County FIPS code) tuple
        each value is a (CBSA FIPS code, CBSA Name) tuple
    """
    logging.info("Beginning CBSA lookup")
    cbsa_lookup = defaultdict(dict)
    download = requests.get(URL)
    decoded_content = download.content.decode('latin-1').encode('utf-8')
    reader = csv.reader(decoded_content.splitlines(), delimiter=',')
    # skip header line
    next(reader)
    for row in reader:
        state_code = row[1]
        fipscounty = row[3][-3:]
        cbsa = row[4]
        cbsaname = row[5]
        cbsa_lookup[state_code][fipscounty] = (cbsa, cbsaname)
    return cbsa_lookup
