import logging

from datasets import ua_cbsa, place_ua, cousub_ua


def city_cleaner(city):
    city = city.lower()
    city = city.replace('.', '')
    city = city.replace('saint', 'st')
    return city


misc_lookup = {
    'HI': {'honolulu': '89770'}
}


class JobCBSAQuerier(object):
    """
    Queries the Core-Based Statistical Area for a job
    """
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
        logging.info('Found %s hits, %s', len(hits), hits)
        return hits
