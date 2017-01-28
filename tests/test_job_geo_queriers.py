from mock import patch
from algorithms.job_geography_queriers.cbsa import JobCBSAQuerier
import unittest

place_ua_lookup = {
    'IL': {'Elgin': '123', 'Gilberts': '234'},
    'MA': {'Elgin': '823'}
}
ua_cbsa_lookup = {
    '123': ['456'],
    '234': ['678'],
    '823': ['987', '876']
}


class CBSATest(unittest.TestCase):
    def setUp(self):
        place_patch = patch('algorithms.job_geography_queriers.cbsa.place_ua', return_value=place_ua_lookup)
        cbsa_patch = patch('algorithms.job_geography_queriers.cbsa.ua_cbsa', return_value=ua_cbsa_lookup)
        self.addCleanup(place_patch.stop)
        self.addCleanup(cbsa_patch.stop)
        place_patch.start()
        cbsa_patch.start()
        self.querier = JobCBSAQuerier()

    def test_querier_one_hit(self):
        sample_job = {
            "description": "We are looking for someone for a job",
            "jobLocation": {
                "@type": "Place",
                "address": {
                    "addressLocality": "Elgin",
                    "addressRegion": "IL",
                    "@type": "PostalAddress"
                }
            },
            "@context": "http://schema.org",
            "alternateName": "Customer Service Representative",
            "datePosted": "2013-03-07",
            "@type": "JobPosting"
        }

        assert self.querier.query(sample_job) == ['456']

    def test_querier_multiple_hits(self):
        sample_job = {
            "description": "We are looking for someone for a job",
            "jobLocation": {
                "@type": "Place",
                "address": {
                    "addressLocality": "Elgin",
                    "addressRegion": "MA",
                    "@type": "PostalAddress"
                }
            },
            "@context": "http://schema.org",
            "alternateName": "Customer Service Representative",
            "datePosted": "2013-03-07",
            "@type": "JobPosting"
        }

        assert self.querier.query(sample_job) == ['987', '876']

    def test_querier_blank(self):
        with self.assertRaises(KeyError):
            self.querier.query({})
