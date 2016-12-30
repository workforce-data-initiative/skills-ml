import logging

from datasets import county_lookup, cbsa_lookup


class JobCBSAQuerier(object):
    def __init__(self):
        self.county_lookup = county_lookup()
        self.cbsa_lookup = cbsa_lookup()

    def query(self, job_posting):
        city = job_posting['jobLocation']['address']['addressLocality']
        state_code = job_posting['jobLocation']['address']['addressRegion']
        logging.debug('Looking up CBSA for %s, %s', city, state_code)
        county_fips, county_name = self.county_lookup[state_code][city]
        cbsa_fips, cbsa_name = self.cbsa_lookup[(state_code, county_fips)]
        logging.debug('Found %s, %s', cbsa_fips, cbsa_name)
        return cbsa_fips, cbsa_name
