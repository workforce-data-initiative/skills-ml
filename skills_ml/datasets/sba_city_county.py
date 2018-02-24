"""Retrieve county lookup tables from the SBA for each state"""
from skills_utils.fs import cache_json
import logging
import requests

URL = 'http://api.sba.gov/geodata/city_county_links_for_state_of/{}.json'

STATE_CODES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]


def _grab_state_data(state_code):
    logging.info('Looking up counties for --%s--', state_code)
    response = requests.get(URL.format(state_code))
    try:
        response = response.json()
        return {
            city['name']: (city['fips_county_cd'], city['county_name'])
            for city in response
        }
    except:
        return {}


@cache_json('county_lookup.json')
def county_lookup():
    """
    Retrieve county lookup tables if they are not already cached

    Returns: (dict) each key is a state, each value is a dict {city_name: (fips_county_code, county_name)}
    """
    return {
        state_code: _grab_state_data(state_code)
        for state_code in STATE_CODES
    }
