"""
Computes geographic representativeness of job postings based on ONET SOC Code
"""
from collections import Counter
from skills_ml.job_postings.geography_queriers.cbsa import JobCBSAFromCrosswalkQuerier


class GeoOccupationRepresentativenessCalculator(object):
    """
    Calculates geographic representativeness of SOC Codes.
    If a job normalizer is given, will attempt to compute SOC codes
    of jobs that have missing SOC codes

    Args:
        geo_querier (skills_ml.job_postings.geography_queriers) An object that can return a CBSA from a job posting
        normalizer (skills_ml.algorithms.occupation_classifiers) An object that can return the SOC code from a job posting

    """
    def __init__(self, geo_querier=None, normalizer=None):
        self.normalizer = normalizer
        self.cbsa_querier = geo_querier or JobCBSAFromCrosswalkQuerier()

    def dataset_distribution(self, job_postings):
        """
        Computes the SOC Code/CBSA distribution of the given job postings
        Args:
            job_postings (iterable) Job postings, each in common schema format
        Returns:
            (collections.Counter)
                The number of job postings for each
                (CBSA FIPS Code, ONET SOC Code) tuple
        """
        dataset_counts = Counter()
        for job_posting in job_postings:
            soc_code = None
            if 'onet_soc_code' in job_posting:
                soc_code = job_posting['onet_soc_code']
            else:
                if self.normalizer is not None:
                    soc_code = self.normalizer.normalize_job_title(job_posting['title'])

            if soc_code:
                cbsa_fips, cbsa_name = self.cbsa_querier.query(job_posting)
                dataset_counts[(cbsa_fips, soc_code)] += 1
        return dataset_counts
