from mock import patch
from algorithms.job_geography_queriers.cbsa import JobCBSAQuerier
import unittest

place_ua_lookup = {
    'IL': {'elgin': '123', 'gilberts': '234'},
    'MA': {'elgin': '823'}
}
cousub_ua_lookup = {
    'ND': {'elgin': '623'}
}
ua_cbsa_lookup = {
    '123': ['456'],
    '234': ['678'],
    '823': ['987', '876'],
    '623': ['345']
}


class CBSATest(unittest.TestCase):
    def setUp(self):
        place_patch = patch('algorithms.job_geography_queriers.cbsa.place_ua', return_value=place_ua_lookup)
        cousub_patch = patch('algorithms.job_geography_queriers.cbsa.cousub_ua', return_value=cousub_ua_lookup)
        cbsa_patch = patch('algorithms.job_geography_queriers.cbsa.ua_cbsa', return_value=ua_cbsa_lookup)
        self.addCleanup(cousub_patch.stop)
        self.addCleanup(place_patch.stop)
        self.addCleanup(cbsa_patch.stop)
        cousub_patch.start()
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
            "@type": "JobPosting",
            "id": 5
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
            "@type": "JobPosting",
            "id": 5
        }

        assert self.querier.query(sample_job) == ['987', '876']

    def test_querier_no_hits(self):
        sample_job = {
            "description": "We are looking for someone for a job",
            "jobLocation": {
                "@type": "Place",
                "address": {
                    "addressLocality": "Elgin",
                    "addressRegion": "TX",
                    "@type": "PostalAddress"
                }
            },
            "@context": "http://schema.org",
            "alternateName": "Customer Service Representative",
            "datePosted": "2013-03-07",
            "@type": "JobPosting",
            "id": 5
        }

        assert self.querier.query(sample_job) == []

    def test_querier_hit_in_alternate(self):
        sample_job = {
            "description": "We are looking for someone for a job",
            "jobLocation": {
                "@type": "Place",
                "address": {
                    "addressLocality": "Elgin",
                    "addressRegion": "ND",
                    "@type": "PostalAddress"
                }
            },
            "@context": "http://schema.org",
            "alternateName": "Customer Service Representative",
            "datePosted": "2013-03-07",
            "@type": "JobPosting",
            "id": 5
        }

        assert self.querier.query(sample_job) == ['345']

    def test_querier_blank(self):
        assert self.querier.query({'id': 5}) == []
