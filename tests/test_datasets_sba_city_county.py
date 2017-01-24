import httpretty
import json
from mock import patch

from datasets.sba_city_county import county_lookup, URL

COUNTY_RESPONSE = json.dumps([
    {
        "county_name": "St. Clair",
        "description": None,
        "feat_class": "Populated Place",
        "feature_id": "4609",
        "fips_class": "C1",
        "fips_county_cd": "163",
        "full_county_name": "St. Clair County",
        "link_title": None,
        "url": "http://www.belleville.net/",
        "name": "Belleville",
        "primary_latitude": "38.52",
        "primary_longitude": "-89.98",
        "state_abbreviation": "IL",
        "state_name": "Illinois"
    }
])


@httpretty.activate
@patch('datasets.sba_city_county.STATE_CODES', ['IL'])
def test_county_lookup():
    httpretty.register_uri(
        httpretty.GET,
        URL.format('IL'),
        body=COUNTY_RESPONSE,
        content_type='application/json'
    )

    lookup = county_lookup.__wrapped__()
    assert lookup['IL'] == {'Belleville': ('163', 'St. Clair')}
