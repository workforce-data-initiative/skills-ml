from mock import patch
from skills_ml.algorithms.job_geography_queriers.cbsa import JobCBSAQuerier
from skills_ml.algorithms.job_geography_queriers.cbsa_from_geocode import JobCBSAFromGeocodeQuerier
import json
import unittest

place_ua_lookup = {
    'IL': {'elgin': '123', 'gilberts': '234'},
    'MA': {'elgin': '823'}
}
cousub_ua_lookup = {
    'ND': {'elgin': '623'}
}
ua_cbsa_lookup = {
    '123': [['456', 'Chicago, IL Metro Area']],
    '234': [['678', 'Rockford, IL Metro Area']],
    '823': [
        ['987', 'Springfield, MA Metro Area'],
        ['876', 'Worcester, MA Metro Area'],
    ],
    '623': [['345', 'Fargo, ND Metro Area']]
}


class CBSATest(unittest.TestCase):
    def setUp(self):
        place_patch = patch(
            'skills_ml.algorithms.job_geography_queriers.cbsa.place_ua',
            return_value=place_ua_lookup
        )
        cousub_patch = patch(
            'skills_ml.algorithms.job_geography_queriers.cbsa.cousub_ua',
            return_value=cousub_ua_lookup
        )
        cbsa_patch = patch(
            'skills_ml.algorithms.job_geography_queriers.cbsa.ua_cbsa',
            return_value=ua_cbsa_lookup
        )
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

        assert self.querier.query(sample_job) == (
            ('456', 'Chicago, IL Metro Area', 'IL'),
        )

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

        assert self.querier.query(sample_job) == (
            ('987', 'Springfield, MA Metro Area', 'MA'),
            ('876', 'Worcester, MA Metro Area', 'MA'),
        )

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

        assert self.querier.query(sample_job) == (
            ('345', 'Fargo, ND Metro Area', 'ND'),
        )

    def test_querier_blank(self):
        assert self.querier.query({'id': 5}) == []


cbsa_results = {
    'Elgin, IL': ['456', 'Chicago, IL Metro Area'],
    'Elgin, TX': None,
}


class CBSAFromGeocodeTest(unittest.TestCase):
    def setUp(self):
        self.querier = JobCBSAFromGeocodeQuerier(cbsa_results=cbsa_results)

    def test_querier_one_hit(self):
        sample_job = json.dumps({
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
        })

        assert self.querier.query(sample_job) == \
            ('456', 'Chicago, IL Metro Area', 'IL')

    def test_querier_hit_no_cbsa(self):
        sample_job = json.dumps({
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
        })

        assert self.querier.query(sample_job) == (None, None, 'TX')

    def test_querier_not_present(self):
        sample_job = json.dumps({
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
        })

        assert self.querier.query(sample_job) == (None, None, 'ND')
