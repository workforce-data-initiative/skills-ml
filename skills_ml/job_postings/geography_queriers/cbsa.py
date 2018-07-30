"""Look up the CBSA for a job posting from a census crosswalk (job location -> Census Place -> Census UA -> Census CBSA)
"""
import logging

from skills_ml.datasets import ua_cbsa, place_ua, cousub_ua
from . import job_posting_search_strings
from .base import JobGeographyQuerier


class JobCBSAFromGeocodeQuerier(JobGeographyQuerier):
    """
    Queries the Core-Based Statistical Area for a job

    This object delegates the CBSA-finding algorithm to a passed-in finder.
    In practice, you can look at the `skills_ml.algorithms.geocoders.cbsa`
    module for an example of how this can be generated.

    Instead, this object focuses on the job posting-centric logic necessary,
    such as converting the job posting to the form needed to use the cache
    and dealing with differents kinds of cache misses.

    Args:
        cbsa_finder (dict) A mapping of geocoding search strings to
            (CBSA FIPS, CBSA Name) tuples
    """

    @property
    def name(self):
        return 'cbsa_from_geocode'

    @property
    def output_columns(self):
        return (
            ('cbsa_fips', 'FIPS code of Core-Based Statistical Area, found by geocoding job location'),
            ('cbsa_name', 'Name of Core-Based Statistical Area, found by geocoding job location')
        )

    def __init__(self, geocoder, cbsa_finder):
        self.geocoder = geocoder
        self.cbsa_finder = cbsa_finder

    def _query(self, job_posting):
        """
        Look up the CBSA from a job posting
        Arguments:
            job_posting (dict) A job posting in common schema json form
        Returns:
            (tuple) (CBSA FIPS Code, CBSA Name)
        """
        search_strings = job_posting_search_strings(job_posting)
        geocode_results = [
            self.geocoder.geocode(search_string) for search_string in search_strings
        ]
        cbsa_results = [
            self.cbsa_finder.query(geocode_result) for geocode_result in geocode_results
        ]

        first_result_with_cbsa = None
        for cbsa_result in cbsa_results:
            if cbsa_result:
                first_result_with_cbsa = cbsa_result
                break

        if not first_result_with_cbsa:
            logging.warning(
                'Returning blank CBSA for %s. Search strings: %s : geocode results: %s',
                job_posting['id'],
                search_strings,
                geocode_results
            )
            return (None, None)

        cbsa_fips, cbsa_name = first_result_with_cbsa

        return (cbsa_fips, cbsa_name)


def city_cleaner(city):
    city = city.lower()
    city = city.replace('.', '')
    city = city.replace('saint', 'st')
    return city


misc_lookup = {
    'HI': {'honolulu': '89770'}
}


class JobCBSAFromCrosswalkQuerier(JobGeographyQuerier):
    """Queries the Core-Based Statistical Area for a job using a census crosswalk

    First looks up a Place or County Subdivision by the job posting's state and city.
    If it finds a result, it will then take the Urbanized Area for that Place or County Subdivison and find CBSAs associated with it.

    Queries return all hits, so there may be multiple CBSAs for a given query.
    """

    name = 'cbsa_census_xwalk'

    # The columns that are returned for each row
    @property
    def output_columns(self):
        return (
            ('cbsa_fips', 'FIPS code of Core-Based Statistical Area, found by census crosswalk'),
            ('cbsa_name', 'Name of Core-Based Statistical Area, found by census crosswalk'),
        )

    def __init__(self):
        self.ua_cbsa = ua_cbsa()
        self.place_ua = place_ua(city_cleaner)
        self.cousub_ua = cousub_ua(city_cleaner)
        self.f = open('missed.txt', 'a')

    def _query(self, job_posting):
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
            return (None, None)

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
            return (None, None)

        if ua_fips not in self.ua_cbsa:
            logging.warning('Could not find %s/%s', state_code, ua_fips)
            return (None, None)

        hits = self.ua_cbsa[ua_fips]
        hits = [tuple(list(hit)) for hit in hits]
        logging.debug('Found %s hits, %s. Returning first', len(hits), hits)
        return hits[0]
