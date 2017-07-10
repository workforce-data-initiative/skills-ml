import csv


class GeoAggregator(object):
    """Aggregates jobs by geography

        job_aggregators (dict of .JobAggregator objects) - The aggregators
            that should accumulate data based on geography and title for each
            job posting
        geo_querier (object) an object that returns a geography of a given job.
    """
    def __init__(self, job_aggregators, geo_querier):
        self.job_aggregators = job_aggregators
        self.geo_querier = geo_querier

    def merge_job_aggregators(self, other_job_aggregators):
        for key, value in other_job_aggregators.items():
            self.job_aggregators[key] += value
        return self

    def process_postings(self, job_postings):
        raise NotImplementedError()

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
