import json
from collections import OrderedDict

from skills_ml.job_postings.aggregate import CountAggregator
from skills_ml.job_postings.aggregate.soc_code import GeoSocAggregator


class FakeCBSAQuerier(object):
    geo_key_names = ('cbsa_fips', 'cbsa_name', 'state_code')

    def query(self, job_listing):
        if json.loads(job_listing)['id'] == 1:
            return ('234', 'A Micro', 'ZY')
        else:
            return ('456', 'A Metro', 'XX')


class FakeSocClassifier(object):
    def classify(self, job_posting):
        if 'frosting' in job_posting:
            return ('11-CAKE.00', 0.5)
        else:
            return ('12-OTHER.00', 0.5)

SAMPLE_JOBS = [
    {'id': 1, 'title': 'Cupcake Ninja', 'description': 'Slicing and dicing frosting'},
    {'id': 2, 'title': 'Regular Ninja', 'description': 'Slicing and dicing enemies'},
    {'id': 3, 'title': 'React Ninja', 'description': 'Slicing and dicing components'},
    {'id': 4, 'title': 'React Ninja', 'description': 'Slicing and dicing and then trashing jQuery'},
]


def test_geo_soc_aggregator_process_postings():
    job_aggregators = OrderedDict(
        count=CountAggregator(),
    )
    aggregator = GeoSocAggregator(
        occupation_classifier=FakeSocClassifier(),
        job_aggregators=job_aggregators,
        geo_querier=FakeCBSAQuerier()
    )
    aggregator.process_postings([json.dumps(job) for job in SAMPLE_JOBS])

    assert aggregator.job_aggregators['count'].group_values == {
        (('456', 'A Metro', 'XX'), '12-OTHER.00'): {'total': 3},
        (('234', 'A Micro', 'ZY'), '11-CAKE.00'): {'total': 1},
    }

    assert aggregator.job_aggregators['count'].rollup == {
        '11-CAKE.00': {'total': 1},
        '12-OTHER.00': {'total': 3},
    }

def test_geo_soc_aggregator_noclassifier():
    job_aggregators = OrderedDict(
        count=CountAggregator(),
    )
    aggregator = GeoSocAggregator(
        job_aggregators=job_aggregators,
        geo_querier=FakeCBSAQuerier()
    )
    aggregator.process_postings([json.dumps(job) for job in SAMPLE_JOBS])

    assert aggregator.job_aggregators['count'].rollup == {
        '99-9999.00': {'total': 4},
    }
