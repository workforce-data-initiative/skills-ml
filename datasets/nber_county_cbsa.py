import unicodecsv as csv
import logging
import requests

from utils.fs import cache_json

URL = 'http://www.nber.org/cbsa-msa-fips-ssa-county-crosswalk/2016/cbsatocountycrosswalk2016.csv'


@cache_json('cbsa_lookup.json')
def cbsa_lookup():
    logging.info("Beginning CBSA lookup")
    cbsa_lookup = {}
    download = requests.get(URL)
    decoded_content = download.content.decode('latin-1').encode('utf-8')
    reader = csv.reader(decoded_content.splitlines(), delimiter=',')
    # skip header line
    next(reader)
    for row in reader:
        state_code = row[1]
        fipscounty = row[3][2:].zfill(3)
        cbsa = row[4]
        cbsaname = row[5]
        cbsa_lookup[(state_code, fipscounty)] = (cbsa, cbsaname)
    return cbsa_lookup
