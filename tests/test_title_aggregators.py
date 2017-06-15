import json
from collections import Counter, OrderedDict
from tempfile import NamedTemporaryFile
import copy
import csv

from skills_ml.algorithms.aggregators import SkillAggregator, CountAggregator, SocCodeAggregator, GivenSocCodeAggregator
from skills_ml.algorithms.aggregators.title import GeoTitleAggregator
from skills_ml.algorithms.string_cleaners import NLPTransforms
from skills_ml.algorithms.skill_extractors.freetext import FakeFreetextSkillExtractor
from skills_ml.algorithms.corpus_creators.basic import SimpleCorpusCreator


class FakeCBSAQuerier(object):
    geo_key_names = ('cbsa_fips', 'cbsa_name', 'state_code')

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


def test_geo_title_aggregator_process_postings():
    job_aggregators = OrderedDict(
        skills=SkillAggregator(
            skill_extractor=FakeFreetextSkillExtractor(
                skills=['slicing', 'dicing', 'jquery']
            ),
            corpus_creator=SimpleCorpusCreator(),
            output_count=2
        ),
        count=CountAggregator(),
    )
    aggregator = GeoTitleAggregator(
        job_aggregators=job_aggregators,
        geo_querier=FakeCBSAQuerier()
    )
    aggregator.process_postings([json.dumps(job) for job in SAMPLE_JOBS])

    assert aggregator.job_aggregators['count'].group_values == {
        (('456', 'A Metro', 'XX'), 'Regular Ninja'): {'total': 1},
        (('456', 'A Metro', 'XX'), 'React Ninja'): {'total': 2},
        (('123', 'Another Metro', 'YY'), 'Cupcake Ninja'): {'total': 1},
        (('234', 'A Micro', 'ZY'), 'Cupcake Ninja'): {'total': 1},
    }

    assert aggregator.job_aggregators['count'].rollup == {
        'Cupcake Ninja': {'total': 1},
        'Regular Ninja': {'total': 1},
        'React Ninja': {'total': 2}
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

def test_geo_title_aggregator_save_counts():
    job_aggregators = OrderedDict()
    job_aggregators['skills'] = SkillAggregator(
        skill_extractor=FakeFreetextSkillExtractor(
            skills=['slicing', 'dicing', 'jquery']
        ),
        corpus_creator=SimpleCorpusCreator(),
        output_count=3,
        output_total=True,
    )
    job_aggregators['count'] = CountAggregator()
    aggregator = GeoTitleAggregator(
        job_aggregators=job_aggregators,
        geo_querier=FakeCBSAQuerier()
    )
    weighted_jobs = copy.deepcopy(SAMPLE_JOBS)
    for job in weighted_jobs:
        job['description'] = 'slicing'
    aggregator.process_postings([json.dumps(job) for job in SAMPLE_JOBS + weighted_jobs])

    with NamedTemporaryFile(mode='w+') as tf:
        aggregator.save_counts(tf.name)
        tf.seek(0)
        reader = csv.reader(tf)
        header_row = next(reader)
        assert header_row == ['cbsa_fips', 'cbsa_name', 'state_code', 'title', 'skills_1', 'skills_2', 'skills_3', 'skills_total', 'count_total']
        data_rows = [row for row in reader]
        expected = [
            ['123', 'Another Metro', 'YY', 'Cupcake Ninja', 'slicing', 'dicing', '', '2', '2'],
            ['234', 'A Micro', 'ZY', 'Cupcake Ninja', 'slicing', 'dicing', '', '2', '2'],
            ['456', 'A Metro', 'XX', 'Regular Ninja', 'slicing', 'dicing', '', '2', '2'],
            ['456', 'A Metro', 'XX', 'React Ninja', 'slicing', 'dicing', 'jquery', '3', '4'],
        ]

        for expected_row in expected:
            assert expected_row in data_rows


def test_geo_title_aggregator_save_rollup():
    job_aggregators = OrderedDict()
    job_aggregators['skills'] = SkillAggregator(
        skill_extractor=FakeFreetextSkillExtractor(
            skills=['slicing', 'dicing', 'jquery']
        ),
        corpus_creator=SimpleCorpusCreator(),
        output_count=2
    )
    job_aggregators['count'] = CountAggregator()
    aggregator = GeoTitleAggregator(
        job_aggregators=job_aggregators,
        geo_querier=FakeCBSAQuerier()
    )
    weighted_jobs = copy.deepcopy(SAMPLE_JOBS)
    for job in weighted_jobs:
        job['description'] = 'slicing'
    aggregator.process_postings([json.dumps(job) for job in SAMPLE_JOBS + weighted_jobs])

    with NamedTemporaryFile(mode='w+') as tf:
        aggregator.save_rollup(tf.name)
        tf.seek(0)
        reader = csv.reader(tf)
        header_row = next(reader)
        assert header_row == ['title', 'skills_1', 'skills_2', 'count_total']
        data_rows = [row for row in reader]
        expected = [
            ['Cupcake Ninja', 'slicing', 'dicing', '2'],
            ['Regular Ninja', 'slicing', 'dicing', '2'],
            ['React Ninja', 'slicing', 'dicing', '4'],
        ]

        for expected_row in expected:
            assert expected_row in data_rows


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
        (('456', 'A Metro', 'XX'), 'regular ninja'): {'total': 1},
        (('456', 'A Metro', 'XX'), 'react ninja'): {'total': 2},
        (('123', 'Another Metro', 'YY'), 'cupcake ninja'): {'total': 1},
        (('234', 'A Micro', 'ZY'), 'cupcake ninja'): {'total': 1}
    }

    assert aggregator.job_aggregators['count'].rollup == {
        'cupcake ninja': {'total': 1},
        'regular ninja': {'total': 1},
        'react ninja': {'total': 2},
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


def test_soc_aggregator():
    class FakeClassifier(object):
        def classify(self, text):
            return ('12-1234.00', 0.5)

    aggregator = SocCodeAggregator(
        occupation_classifier=FakeClassifier(),
        corpus_creator=SimpleCorpusCreator()
    )

    aggregate = sum(map(aggregator.value, SAMPLE_JOBS), Counter())
    assert aggregate == Counter({'12-1234.00': len(SAMPLE_JOBS)})


def test_given_soc_aggregator():
    aggregator = GivenSocCodeAggregator()
    aggregate = sum(map(aggregator.value, SAMPLE_JOBS), Counter())
    assert aggregate == {'99-9999.00': len(SAMPLE_JOBS)}

    weighted_jobs = copy.deepcopy(SAMPLE_JOBS)
    for job in weighted_jobs:
        if job['id'] == 1:
            job['onet_soc_code'] = '23-1234.00'
        else:
            job['onet_soc_code'] = '12-1234.00'
    aggregate = sum(map(aggregator.value, weighted_jobs), Counter())
    assert aggregate == {'23-1234.00': 1, '12-1234.00': 3}
