"""Look up the CBSA for a job posting from a census crosswalk (job location -> Census Place -> Census UA -> Census CBSA)
"""
import logging

from skills_ml.datasets import ua_cbsa, place_ua, cousub_ua


def city_cleaner(city):
    city = city.lower()
    city = city.replace('.', '')
    city = city.replace('saint', 'st')
    return city


misc_lookup = {
    'HI': {'honolulu': '89770'}
}


class JobCBSAQuerier(object):
    """Queries the Core-Based Statistical Area for a job using a census crosswalk

    First looks up a Place or County Subdivision by the job posting's state and city.
    If it finds a result, it will then take the Urbanized Area for that Place or County Subdivison and find CBSAs associated with it.

    Queries return all hits, so there may be multiple CBSAs for a given query.
    """

    # The columns that are returned for each row
    geo_key_names = ('cbsa_fips', 'cbsa_name', 'state_code')

    def __init__(self):
        self.ua_cbsa = ua_cbsa()
        self.place_ua = place_ua(city_cleaner)
        self.cousub_ua = cousub_ua(city_cleaner)
        self.f = open('missed.txt', 'a')

    def query(self, job_posting):
        """
        Look up the CBSA from a job posting
        Arguments:
            job_posting (dict) in common schema format
        Returns:
            (tuple) (CBSA Fips Code, CBSA Name)
        """
        city = job_posting\
            .get('jobLocation', {})\
            .get('address', {})\
            .get('addressLocality', None)
        if city:
            city = city_cleaner(city)
        else:
            logging.warning(
                'Returning blank CBSA for %s as no city was given',
                job_posting['id']
            )
            return []

        state_code = job_posting['jobLocation']['address']['addressRegion']
        logging.debug('Looking up CBSA for %s, %s', city, state_code)

        ua_fips = None
        for lookup in [self.place_ua, self.cousub_ua, misc_lookup]:
            ua_fips = lookup.get(state_code, {}).get(city, None)
            if ua_fips:
                break
        if not ua_fips:
            logging.warning('Could not find %s/%s', state_code, city)
            self.f.write('{}/{}\n'.format(state_code, city))
            return []

        if ua_fips not in self.ua_cbsa:
            logging.warning('Could not find %s/%s', state_code, ua_fips)
            return []

        hits = self.ua_cbsa[ua_fips]
        hits = [tuple(list(hit) + [state_code]) for hit in hits]
        logging.debug('Found %s hits, %s', len(hits), hits)
        return tuple(hits)
