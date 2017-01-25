import logging

from datasets import county_lookup, cbsa_lookup


class JobCBSAQuerier(object):
    """
    Queries the Core-Based Statistical Area for a job
    """
    def __init__(self):
        self.county_lookup = county_lookup()
        self.cbsa_lookup = cbsa_lookup()

    def query(self, job_posting):
        """
        Look up the CBSA from a job posting
        Arguments:
            job_posting (dict) in common schema format
        Returns:
            (tuple) (CBSA Fips Code, CBSA Name)
        """
        city = job_posting['jobLocation']['address']['addressLocality']
        state_code = job_posting['jobLocation']['address']['addressRegion']
        logging.debug('Looking up CBSA for %s, %s', city, state_code)
        if state_code not in self.county_lookup or city not in self.county_lookup[state_code]:
            logging.warning('Could not find %s/%s', state_code, city)
            return None, None
        county_fips, county_name = self.county_lookup[state_code][city]
        county_fips = county_fips.zfill(3)
        if state_code not in self.cbsa_lookup or county_fips not in self.cbsa_lookup[state_code]:
            logging.warning('Could not find %s/%s', state_code, county_fips)
            return None, None
        cbsa_fips, cbsa_name = self.cbsa_lookup[state_code][county_fips]
        logging.info('Found %s, %s', cbsa_fips, cbsa_name)
        return cbsa_fips, cbsa_name
