"""Aggregates job titles
"""
import logging
import json
from skills_ml.algorithms.aggregators.geo import GeoAggregator


class GeoTitleAggregator(GeoAggregator):
    """Aggregates job titles by geography

    Args:
        title_cleaner (function) a function that cleans a given job title
    """
    def __init__(
        self,
        title_cleaner=None,
        *args,
        **kwargs
    ):
        super(GeoTitleAggregator, self).__init__(*args, **kwargs)
        self.title_cleaner = title_cleaner or (lambda s: s)
        self.job_key_name = 'title'

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
            geography_hit = self.geo_querier.query(line)

            for aggregator in self.job_aggregators.values():
                aggregator.accumulate(
                    job_posting=job_posting,
                    job_key=job_title,
                    groups=(geography_hit,)
                )
            if i % 1000 == 0:
                logging.info('Aggregated %s job postings', i)
