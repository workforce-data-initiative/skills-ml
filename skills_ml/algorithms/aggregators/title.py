"""Aggregates job titles
"""
import logging
import json
import csv
from skills_ml.algorithms.job_geography_queriers.cbsa import JobCBSAQuerier


class GeoTitleAggregator(object):
    """Aggregates job titles by geography

    Args:
        job_aggregators (dict of .JobAggregator objects) - The aggregators
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
            geography_hits = self.geo_querier.query(job_posting)

            for aggregator in self.job_aggregators.values():
                aggregator.accumulate(
                    job_posting=job_posting,
                    job_key=job_title,
                    groups=geography_hits
                )
            if i % 1000 == 0:
                logging.info('Aggregated %s job postings', i)

    def save_counts(self, outfilename):
        with open(outfilename, 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            ordered_job_aggregators = []
            header_row = list(self.geo_querier.geo_key_names)\
                + [self.job_key_name]
            for agg_prefix, job_aggregator in self.job_aggregators.items():
                header_row += job_aggregator.output_header_row(agg_prefix)
                ordered_job_aggregators.append(job_aggregator)
            writer.writerow(header_row)

            # all job aggregators should have the same set of keys,
            # so we can just take the keys from the first aggregator
            first_agg = ordered_job_aggregators[0]
            for full_key, values in first_agg.group_values.items():
                group_key, job_key = full_key
                row = group_key + (job_key,)
                for agg in ordered_job_aggregators:
                    row += tuple(agg.group_outputs(full_key))
                writer.writerow(row)

    def save_rollup(self, outfilename):
        with open(outfilename, 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            ordered_job_aggregators = []
            header_row = [self.job_key_name]
            for agg_prefix, job_aggregator in self.job_aggregators.items():
                header_row += job_aggregator.output_header_row(agg_prefix)
                ordered_job_aggregators.append(job_aggregator)
            writer.writerow(header_row)

            # all job aggregators should have the same set of keys,
            # so we can just take the keys from the first aggregator
            first_agg = ordered_job_aggregators[0]
            for job_key, values in first_agg.rollup.items():
                row = [job_key]
                for agg in ordered_job_aggregators:
                    row += tuple(agg.rollup_outputs(job_key))
                writer.writerow(row)
