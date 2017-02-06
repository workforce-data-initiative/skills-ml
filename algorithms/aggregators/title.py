"""Aggregates job titles
"""
import logging
import json
from collections import Counter
from algorithms.job_geography_queriers.cbsa import JobCBSAQuerier


class GeoTitleAggregator(object):
    """Aggregates job titles by geography

    Args:
        geo_querier (object) an object that returns a geography of a given job
            Optional, defaults to JobCBSAQuerier
    """
    def __init__(self, geo_querier=None, title_cleaner=None):
        self.title_cleaner = title_cleaner or (lambda s: s)
        self.geo_querier = geo_querier or JobCBSAQuerier()

    def counts(self, job_postings):
        """
        Computes the title/CBSA distribution of the given job postings
        Args:
            job_postings (iterable) Job postings, each in common schema format
        Returns:
            (collections.Counter)
                The number of job postings for each
                (CBSA FIPS Code, Job Title) tuple
        """
        counts = Counter()
        title_rollup = Counter()
        for line in job_postings:
            job_posting = json.loads(line)
            job_title = self.title_cleaner(job_posting['title'])
            title_rollup[job_title] += 1
            hits = self.geo_querier.query(job_posting)
            for cbsa_fips in hits:
                counts[(cbsa_fips, job_title)] += 1
                logging.info(
                    '%s, %s, %s',
                    cbsa_fips,
                    job_title,
                    counts[(cbsa_fips, job_title)]
                )
        return counts, title_rollup
