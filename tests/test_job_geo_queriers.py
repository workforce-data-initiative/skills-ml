from mock import patch
from algorithms.job_geography_queriers.cbsa import JobCBSAQuerier
import unittest

county_lookup = {
    'IL': {'Elgin': ('123', 'Cook'), 'Gilberts': ('234', 'Kane')},
    'MA': {'Elgin': ('823', 'Sussex')}
}
cbsa_lookup = {
    ('IL', '123'): ('456', 'Chicago, IL'),
    ('IL', '234'): ('678', 'Rockford, IL'),
    ('MA', '823'): ('987', 'Boston, MA')
}

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


class CBSATest(unittest.TestCase):
    @patch('algorithms.job_geography_queriers.cbsa.county_lookup')
    @patch('algorithms.job_geography_queriers.cbsa.cbsa_lookup')
    def test_cbsa_querier(self, cbsa_mock, county_mock):
        county_mock.return_value = county_lookup
        cbsa_mock.return_value = cbsa_lookup
        querier = JobCBSAQuerier()
        assert querier.query(sample_job) == ('456', 'Chicago, IL')
        with self.assertRaises(KeyError):
            querier.query({})
