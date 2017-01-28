import logging

from datasets import ua_cbsa, place_ua


class JobCBSAQuerier(object):
    """
    Queries the Core-Based Statistical Area for a job
    """
    def __init__(self):
        self.ua_cbsa = ua_cbsa()
        self.place_ua = place_ua()

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
        if state_code not in self.place_ua or city not in self.place_ua[state_code]:
            logging.warning('Could not find %s/%s', state_code, city)
            return []
        ua_fips = self.place_ua[state_code][city]
        if ua_fips not in self.ua_cbsa:
            logging.warning('Could not find %s/%s', state_code, ua_fips)
            return []
        hits = self.ua_cbsa[ua_fips]
        logging.info('Found %s hits, %s', len(hits), hits)
        return hits
