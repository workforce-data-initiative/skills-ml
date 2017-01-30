import json

from algorithms.aggregators.title import GeoTitleAggregator
from utils.nlp import NLPTransforms


class FakeCBSAQuerier(object):
    def query(self, job_listing):
        if job_listing['id'] == 1:
            return ['123', '234']
        else:
            return ['456']

SAMPLE_JOBS = [
    {'id': 1, 'title': 'Cupcake Ninja'},
    {'id': 2, 'title': 'Regular Ninja'},
    {'id': 3, 'title': 'React Ninja'},
    {'id': 4, 'title': 'React Ninja'},
]


def test_geo_title_aggregator():
    aggregator = GeoTitleAggregator(
        geo_querier=FakeCBSAQuerier(),
    )
    counts, title_rollup = aggregator.counts(
        [json.dumps(job) for job in SAMPLE_JOBS]
    )
    assert counts == {
        ('456', 'Regular Ninja'): 1,
        ('456', 'React Ninja'): 2,
        ('123', 'Cupcake Ninja'): 1,
        ('234', 'Cupcake Ninja'): 1
    }

    assert title_rollup == {
        'Cupcake Ninja': 1,
        'Regular Ninja': 1,
        'React Ninja': 2,
    }


def test_geo_title_aggregator_with_cleaning():
    nlp = NLPTransforms()
    aggregator = GeoTitleAggregator(
        geo_querier=FakeCBSAQuerier(),
        title_cleaner=nlp.lowercase_strip_punc
    )
    counts, title_rollup = aggregator.counts(
        [json.dumps(job) for job in SAMPLE_JOBS]
    )
    assert counts == {
        ('456', 'regular ninja'): 1,
        ('456', 'react ninja'): 2,
        ('123', 'cupcake ninja'): 1,
        ('234', 'cupcake ninja'): 1
    }

    assert title_rollup == {
        'cupcake ninja': 1,
        'regular ninja': 1,
        'react ninja': 2,
    }
