"""Aggregates job titles
"""
import logging
import json
from skills_ml.algorithms.job_geography_queriers.cbsa import JobCBSAQuerier


class GeoTitleAggregator(object):
    """Aggregates job titles by geography

    Args:
        job_aggregators (list of .JobAggregator objects) - The aggregators
            that should accumulate data based on geography and title for each
            job posting
        geo_querier (object) an object that returns a geography of a given job
            Optional, defaults to JobCBSAQuerier
        title_cleaner (function) a function that cleans a given job title
    """
    def __init__(
        self,
        job_aggregators,
        geo_querier=None,
        title_cleaner=None,
    ):
        self.job_aggregators = job_aggregators
        self.title_cleaner = title_cleaner or (lambda s: s)
        self.geo_querier = geo_querier or JobCBSAQuerier()

    def process_postings(self, job_postings):
        """
        Computes the title/CBSA distribution of the given job postings
        Args:
            job_postings (iterable) Job postings, each in common schema format

        When complete, the aggregators in self.job_aggregators will be updated
        with data from the job postings
        """
        for i, line in enumerate(job_postings):
            job_posting = json.loads(line)
            job_title = self.title_cleaner(job_posting['title'])
            geography_hits = self.geo_querier.query(job_posting)

            for aggregator in self.job_aggregators.values():
                aggregator.accumulate(
                    job_posting=job_posting,
                    job_key=job_title,
                    groups=geography_hits
                )
            if i % 1000 == 0:
                logging.info('Aggregated %s job postings', i)
