import json
import logging
from skills_ml.algorithms.geocoders import job_posting_search_string


class JobCBSAFromGeocodeQuerier(object):
    """
    Queries the Core-Based Statistical Area for a job

    This object delegates the CBSA-finding algorithm to a passed-in cache.
    In practice, you can look at the `skills_ml.algorithms.geocoders.cbsa`
    module for an example of how this can be generated.

    Instead, this object focuses on the job posting-centric logic necessary,
    such as converting the job posting to the form needed to use the cache
    and dealing with differents kinds of cache misses.

    Args:
        cbsa_results (dict) Geocoding search strings mapping to
            (CBSA FIPS, CBSA Name) tuples
    """

    # The columns that are returned for each row
    geo_key_names = ('cbsa_fips', 'cbsa_name', 'state_code')

    def __init__(self, cbsa_results):
        self.cbsa_results = cbsa_results

    def query(self, job_posting):
        """
        Look up the CBSA from a job posting
        Arguments:
            job_posting (string) A job posting in common schema json form
        Returns:
            (tuple) (CBSA FIPS Code, CBSA Name, State)
        """
        post = json.loads(job_posting)
        state_code = post\
            .get('jobLocation', {})\
            .get('address', {})\
            .get('addressRegion', None)

        if not state_code:
            logging.warning(
                'Returning blank CBSA for %s, no state found',
                post['id']
            )
            return (None, None, None)

        search_string = job_posting_search_string(job_posting)
        if search_string not in self.cbsa_results:
            logging.warning(
                'Returning blank CBSA for %s, %s not found in cache',
                post['id'],
                search_string
            )
            return (None, None, state_code)

        cbsa_result = self.cbsa_results[search_string]
        if not cbsa_result:
            logging.warning(
                'Returning blank CBSA for %s, %s found in cache as outside CBSA',
                post['id'],
                search_string
            )
            return (None, None, state_code)

        cbsa_fips, cbsa_name = cbsa_result

        return (cbsa_fips, cbsa_name, state_code)
