import json

from skills_ml.algorithms.aggregators import SkillAggregator, CountAggregator
from skills_ml.algorithms.aggregators.title import GeoTitleAggregator
from skills_ml.algorithms.string_cleaners import NLPTransforms
from skills_ml.algorithms.skill_extractors.freetext import FakeFreetextSkillExtractor
from skills_ml.algorithms.corpus_creators.basic import SimpleCorpusCreator


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
    {'id': 1, 'title': 'Cupcake Ninja', 'description': 'Slicing and dicing frosting'},
    {'id': 2, 'title': 'Regular Ninja', 'description': 'Slicing and dicing enemies'},
    {'id': 3, 'title': 'React Ninja', 'description': 'Slicing and dicing components'},
    {'id': 4, 'title': 'React Ninja', 'description': 'Slicing and dicing and then trashing jQuery'},
]


def test_geo_title_aggregator():
    job_aggregators = {
        'skills': SkillAggregator(
            skill_extractor=FakeFreetextSkillExtractor(
                skills=['slicing', 'dicing', 'jquery']
            ),
            corpus_creator=SimpleCorpusCreator()
        ),
        'count': CountAggregator(),
    }
    aggregator = GeoTitleAggregator(
        job_aggregators=job_aggregators,
        geo_querier=FakeCBSAQuerier()
    )
    aggregator.process_postings([json.dumps(job) for job in SAMPLE_JOBS])

    assert aggregator.job_aggregators['count'].group_values == {
        (('456', 'A Metro', 'XX'), 'Regular Ninja'): 1,
        (('456', 'A Metro', 'XX'), 'React Ninja'): 2,
        (('123', 'Another Metro', 'YY'), 'Cupcake Ninja'): 1,
        (('234', 'A Micro', 'ZY'), 'Cupcake Ninja'): 1
    }

    assert aggregator.job_aggregators['count'].rollup == {
        'Cupcake Ninja': 1,
        'Regular Ninja': 1,
        'React Ninja': 2,
    }

    assert aggregator.job_aggregators['skills'].group_values == {
        (('123', 'Another Metro', 'YY'), 'Cupcake Ninja'): {'slicing': 1, 'dicing': 1},
        (('234', 'A Micro', 'ZY'), 'Cupcake Ninja'): {'slicing': 1, 'dicing': 1},
        (('456', 'A Metro', 'XX'), 'Regular Ninja'): {'slicing': 1, 'dicing': 1},
        (('456', 'A Metro', 'XX'), 'React Ninja'): {'slicing': 2, 'dicing': 2, 'jquery': 1}
    }

    assert aggregator.job_aggregators['skills'].rollup == {
        'Cupcake Ninja': {'slicing': 1, 'dicing': 1},
        'Regular Ninja': {'slicing': 1, 'dicing': 1},
        'React Ninja': {'slicing': 2, 'dicing': 2, 'jquery': 1}
    }


def test_geo_title_aggregator_with_cleaning():
    nlp = NLPTransforms()
    job_aggregators = {
        'skills': SkillAggregator(
            skill_extractor=FakeFreetextSkillExtractor(
                skills=['slicing', 'dicing', 'jquery']
            ),
            corpus_creator=SimpleCorpusCreator()
        ),
        'count': CountAggregator(),
    }
    aggregator = GeoTitleAggregator(
        job_aggregators=job_aggregators,
        geo_querier=FakeCBSAQuerier(),
        title_cleaner=nlp.lowercase_strip_punc,
    )
    aggregator.process_postings([json.dumps(job) for job in SAMPLE_JOBS])

    assert aggregator.job_aggregators['count'].group_values == {
        (('456', 'A Metro', 'XX'), 'regular ninja'): 1,
        (('456', 'A Metro', 'XX'), 'react ninja'): 2,
        (('123', 'Another Metro', 'YY'), 'cupcake ninja'): 1,
        (('234', 'A Micro', 'ZY'), 'cupcake ninja'): 1
    }

    assert aggregator.job_aggregators['count'].rollup == {
        'cupcake ninja': 1,
        'regular ninja': 1,
        'react ninja': 2,
    }

    assert aggregator.job_aggregators['skills'].group_values == {
        (('123', 'Another Metro', 'YY'), 'cupcake ninja'): {'slicing': 1, 'dicing': 1},
        (('234', 'A Micro', 'ZY'), 'cupcake ninja'): {'slicing': 1, 'dicing': 1},
        (('456', 'A Metro', 'XX'), 'regular ninja'): {'slicing': 1, 'dicing': 1},
        (('456', 'A Metro', 'XX'), 'react ninja'): {'slicing': 2, 'dicing': 2, 'jquery': 1}
    }

    assert aggregator.job_aggregators['skills'].rollup == {
        'cupcake ninja': {'slicing': 1, 'dicing': 1},
        'regular ninja': {'slicing': 1, 'dicing': 1},
        'react ninja': {'slicing': 2, 'dicing': 2, 'jquery': 1}
    }
