"""Geocoders, with caching and throttling"""
import json
import logging
import time
import geocoder
import boto
import traceback

from skills_utils.s3 import split_s3_path, S3BackedJsonDict



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
        cache_s3_path,
        geocode_func=geocoder.osm,
        sleep_time=1,
    ):
        self.cache_s3_path = cache_s3_path
        self.geocode_func = geocode_func
        self.sleep_time = sleep_time
        self.cache = S3BackedJsonDict(path=self.cache_s3_path)

    def retrieve_from_cache(self, search_strings):
        """Retrieve a saved geocode result from the cache if it exists

        Usable in parallel, since it will not perform geocoding on its own.
        This means that you should make code that calls this dependent
        on the geocoding for the job posting being completed

        Args:
            job_posting (string) A job posting in schema.org/JobPosting json form
        Returns: (string) The geocoding result, or None if none is available
        """
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
        if search_string not in self.cache:
            logging.info('%s not found in cache, geocoding', search_string)
            self.cache[search_string] = self.geocode_func(search_string).json
            time.sleep(self.sleep_time)
        return self.cache[search_string]

    def save(self):
        """Save the geocoding cache to S3"""
        self.cache.save()
        logging.info(
            'Successfully saved geocoding cache to %s',
            self.cache_s3_path
        )

    @property
    def all_cached_geocodes(self):
        """Return the contents of the geocoding cache

        Returns: (dict) search strings mapping to their (dict) geocoded results
        """
        return self.cache

    def geocode_search_strings_and_save(self, search_strings, save_every=100000):
        """Geocode job postings and save the results to S3

        Args:
            search_strings (iterable) Strings to geocode
            save_every (int) How frequently to defensively save the cache
                Defaults to every 100000 job postings
        """
        processed = 0
        skipped = 0
        try:
            for i, search_string in enumerate(search_strings):
                self.geocode(search_string)
                processed += 1

        except Exception:
            logging.error('Quitting geocoding due to %s', traceback.format_exc())

        logging.info('Geocoded %s', processed)
        self.save()
