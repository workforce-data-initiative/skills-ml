"""Geocoders, with caching and throttling"""
import json
import logging
import time
import geocoder
import boto
import us
import traceback

from skills_utils.s3 import split_s3_path

STATE_NAME_LOOKUP = us.states.mapping('abbr', 'name')


def job_posting_search_strings(job_posting):
    """Convert a job posting to a geocode-ready search string

    Includes city and state if present, or just city

    Args:
        job_posting (string) A job posting in schema.org/JobPosting json form

    Returns: (string) A geocode-ready search string
    """
    location = json.loads(job_posting).get('jobLocation', None)
    if not location:
        return []
    locality = location.get('address', {}).get('addressLocality', None)
    region = location.get('address', {}).get('addressRegion', None)
    if locality and region:
        # lookup state name, if it's not there just use whatever they typed

        lookups = ['{}, {}'.format(locality, region)]
        formatted_region = STATE_NAME_LOOKUP.get(region, None)
        if formatted_region:
            lookups.append('{}, {}'.format(locality, formatted_region))
        return lookups
    elif locality:
        return [locality]
    else:
        return []


class S3CachedGeocoder(object):
    """Geocoder that uses S3 as a cache.

    Args:
        s3_conn (boto.s3.connection) an s3 connection
        cache_s3_path (string) path (including bucket) to the json cache on s3
        geocode_func (function) a function that geocodes a given search string
            defaults to the OSM geocoder provided by the geocode library
        sleep_time (int) The time, in seconds, between geocode calls
    """
    def __init__(
        self,
        s3_conn,
        cache_s3_path,
        geocode_func=geocoder.osm,
        sleep_time=1,
    ):
        self.s3_conn = s3_conn
        self.cache_s3_path = cache_s3_path
        self.geocode_func = geocode_func
        self.sleep_time = sleep_time
        self.cache = None

    @property
    def _key(self):
        bucket_name, path = split_s3_path(self.cache_s3_path)
        return boto.s3.key.Key(
            bucket=self.s3_conn.get_bucket(bucket_name),
            name=path
        )

    def _load(self):
        try:
            cache_json = json.loads(self._key.get_contents_as_string().decode('utf-8'))
            self.cache = cache_json
            # results for a new file can be None instead of empty dict,
            # so explicitly handle that case
            if not self.cache:
                self.cache = {}

        except boto.exception.S3ResponseError as e:
            logging.warning(
                'Geocoder cachefile load failed with exception %s,' +
                'will overwrite', e
            )
            self.cache = {}
        assert isinstance(self.cache, dict)

    def retrieve_from_cache(self, job_posting):
        """Retrieve a saved geocode result from the cache if it exists

        Usable in parallel, since it will not perform geocoding on its own.
        This means that you should make code that calls this dependent
        on the geocoding for the job posting being completed

        Args:
            job_posting (string) A job posting in schema.org/JobPosting json form
        Returns: (string) The geocoding result, or None if none is available
        """
        if not self.cache:
            self._load()
        search_strings = job_posting_search_strings(job_posting)
        cache_results = [
            self.cache.get(search_string, None)
            for search_string in search_strings
        ]
        return cache_results

    def geocode(self, search_string):
        """Geocodes a single search string

        First checks in cache to see if the search string has been geocoded
        If the geocoding function is called, the process will sleep afterwards

        Warning: Not to be used in parallel!
        To respect throttling limits, it would be dangerous to try and run more
        then one of these against a geocoder service. As a result, the code
        assumes that when it comes time to save, there have been no
        external changes to the file on S3 worth keeping.

        Args:
            search_string (string) A search query to send to the geocoder
        Returns: (string) The geocoding result
        """
        if not self.cache:
            self._load()
        if search_string not in self.cache:
            logging.info('%s not found in cache, geocoding', search_string)
            self.cache[search_string] = self.geocode_func(search_string).json
            time.sleep(self.sleep_time)
        return self.cache[search_string]

    def save(self):
        """Save the geocoding cache to S3"""
        self._key.set_contents_from_string(json.dumps(self.cache))
        logging.info(
            'Successfully saved geocoding cache to %s',
            self.cache_s3_path
        )

    @property
    def all_cached_geocodes(self):
        """Return the contents of the geocoding cache

        Returns: (dict) search strings mapping to their (dict) geocoded results
        """
        if not self.cache:
            self._load()
        return self.cache

    def geocode_job_postings_and_save(self, job_postings, save_every=100000):
        """Geocode job postings and save the results to S3

        Args:
            job_postings (iterable) Job postings in common schema format
            save_every (int) How frequently to defensively save the cache
                Defaults to every 100000 job postings
        """
        skipped = 0
        processed = 0
        try:
            for i, job_posting in enumerate(job_postings):
                search_strings = job_posting_search_strings(job_posting)
                if not search_strings:
                    skipped += 1
                    continue
                for search_string in search_strings:
                    self.geocode(search_string)
                processed += 1
                if i % save_every == 0:
                    logging.info(
                        'Geocoding update: %s total, %s cache size',
                        i,
                        len(self.cache.keys())
                    )
                    self.save()

        except Exception:
            logging.error('Quitting geocoding due to %s', traceback.format_exc())

        logging.info(
            'Geocoded %s, skipped %s due to lack of location',
            processed,
            skipped
        )
        self.save()
