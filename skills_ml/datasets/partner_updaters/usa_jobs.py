"""Update raw job postings from the USAJobs API"""

import logging
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class USAJobsUpdater(object):
    base_url = 'https://data.usajobs.gov/api/Search'

    def __init__(self, auth_key, key_email, session=None):
        self.auth_key = auth_key
        self.key_email = key_email
        self.session = session
        self._all_postings = None
        if not self.session:
            self.session = self._default_session()

    @property
    def headers(self):
        return {
            'Host': 'data.usajobs.gov',
            'User-Agent': self.key_email,
            'Authorization-Key': self.auth_key
        }

    def _default_session(self):
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[502, 503, 504]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def _get_page_json(self, page_num):
        print('Retrieving page {}'.format(page_num))
        response = self.session.get(
            self.base_url,
            params={'ResultsPerPage': 500, 'Page': page_num},
            headers=self.headers
        )
        return response.json()

    def _search_result(self, page_num):
        logging.info('Retrieving page %s', page_num)
        return self._get_page_json(page_num)['SearchResult']

    def _result_items(self, result):
        return result['SearchResultItems']

    def _number_of_pages(self, result):
        return int(result['UserArea']['NumberOfPages'])

    @property
    def all_postings(self):
        if not self._all_postings:
            result = self._search_result(1)
            all_postings = self._result_items(result)
            pages = self._number_of_pages(result)
            for page in range(2, pages+1):
                result = self._search_result(page)
                logging.info(
                    'count on page %s: %s',
                    page,
                    result['SearchResultCount']
                )
                logging.info(
                    'count overall: %s',
                    result['SearchResultCountAll']
                )
                all_postings += self._result_items(result)
            logging.info('%s total postings found', len(all_postings))
            self._all_postings = all_postings
        return self._all_postings

    def deduplicated_postings(self):
        lookup = {}
        for job in self.all_postings:
            ctlnum = job['MatchedObjectId']
            if ctlnum in lookup:
                logging.warning('Duplicate found, replacing %s', ctlnum)
            lookup[ctlnum] = job['MatchedObjectDescriptor']
        logging.info('%s deduplicated postings found', len(lookup.keys()))
        return lookup
