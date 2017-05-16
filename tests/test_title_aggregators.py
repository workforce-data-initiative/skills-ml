import json

from skills_ml.algorithms.aggregators.title import GeoTitleAggregator
from skills_ml.algorithms.string_cleaners import NLPTransforms


class FakeCBSAQuerier(object):
    def query(self, job_listing):
        if job_listing['id'] == 1:
            return [
                ('123', 'Another Metro', 'YY'),
                ('234', 'A Micro', 'ZY')
            ]
        else:
            return [('456', 'A Metro', 'XX')]

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
        ('456', 'A Metro', 'XX', 'Regular Ninja'): 1,
        ('456', 'A Metro', 'XX', 'React Ninja'): 2,
        ('123', 'Another Metro', 'YY', 'Cupcake Ninja'): 1,
        ('234', 'A Micro', 'ZY', 'Cupcake Ninja'): 1
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
        ('456', 'A Metro', 'XX', 'regular ninja'): 1,
        ('456', 'A Metro', 'XX', 'react ninja'): 2,
        ('123', 'Another Metro', 'YY', 'cupcake ninja'): 1,
        ('234', 'A Micro', 'ZY', 'cupcake ninja'): 1
    }

    assert title_rollup == {
        'cupcake ninja': 1,
        'regular ninja': 1,
        'react ninja': 2,
    }
