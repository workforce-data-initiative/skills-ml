import json
from collections import Counter, OrderedDict
from tempfile import NamedTemporaryFile
import copy
import csv
from multiprocessing import Pool

from skills_utils.iteration import Batch

from skills_ml.job_postings.aggregate import \
    SkillAggregator,\
    OccupationScopedSkillAggregator,\
    CountAggregator,\
    SocCodeAggregator,\
    GivenSocCodeAggregator
from skills_ml.job_postings.aggregate.title import GeoTitleAggregator
from skills_ml.algorithms.string_cleaners import NLPTransforms
from skills_ml.algorithms.skill_extractors import \
    ExactMatchSkillExtractor, SocScopedExactMatchSkillExtractor
from skills_ml.job_postings.corpora.basic import SimpleCorpusCreator


class FakeExactMatchSkillExtractor(ExactMatchSkillExtractor):
    """A skill extractor that takes a list of skills
    instead of reading from a filename
    """
    def __init__(self, skills):
        """
        Args:
            skills (list) skill names that the extractor should use
        """
        self.skills = skills
        super().__init__('')

    def _skills_lookup(self):
        return set(self.skills)


class FakeOccupationScopedSkillExtractor(SocScopedExactMatchSkillExtractor):
    """A skill extractor that takes a list of skills
    instead of reading from a filename
    """
    def __init__(self, skills):
        """
        Args:
            skills (dict) soc codes as keys, lists of skill names as values
        """
        self.skills = skills
        super().__init__('')

    def _skills_lookup(self):
        return self.skills


class FakeCBSAQuerier(object):
    geo_key_names = ('cbsa_fips', 'cbsa_name', 'state_code')

    def query(self, job_listing):
        if json.loads(job_listing)['id'] == 1:
            return ('123', 'Another Metro', 'YY')
        else:
            return ('456', 'A Metro', 'XX')

SAMPLE_JOBS = [
    {
        'id': 1,
        'title': 'Cupcake Ninja',
        'description': 'Slicing and dicing frosting',
        'onet_soc_code': '23-1234.00',
    },
    {
        'id': 2,
        'title': 'Regular Ninja',
        'description': 'Slicing and dicing enemies',
        'onet_soc_code': '12-1234.00',
    },
    {
        'id': 3,
        'title': 'React Ninja',
        'description': 'Slicing and dicing components',
        'onet_soc_code': '12-1234.00',
    },
    {
        'id': 4,
        'title': 'React Ninja',
        'description': 'Slicing and dicing and then trashing jQuery',
        'onet_soc_code': '12-1234.00',
    },
]


def build_basic_geo_title_aggregator():
    job_aggregators = OrderedDict(
        skills=OccupationScopedSkillAggregator(
            skill_extractor=FakeOccupationScopedSkillExtractor(
                skills={
                    '12-1234.00': ['slicing', 'dicing', 'jquery'],
                    '23-1234.00': ['slicing', 'dicing', 'jquery'],
                }
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
    return aggregator


def basic_assertions(aggregator):
    assert aggregator.job_aggregators['count'].group_values == {
        (('456', 'A Metro', 'XX'), 'Regular Ninja'): {'total': 1},
        (('456', 'A Metro', 'XX'), 'React Ninja'): {'total': 2},
        (('123', 'Another Metro', 'YY'), 'Cupcake Ninja'): {'total': 1},
    }

    assert aggregator.job_aggregators['count'].rollup == {
        'Cupcake Ninja': {'total': 1},
        'Regular Ninja': {'total': 1},
        'React Ninja': {'total': 2}
    }

    assert aggregator.job_aggregators['skills'].group_values == {
        (('123', 'Another Metro', 'YY'), 'Cupcake Ninja'):
            {'slicing': 1, 'dicing': 1},
        (('456', 'A Metro', 'XX'), 'Regular Ninja'):
            {'slicing': 1, 'dicing': 1},
        (('456', 'A Metro', 'XX'), 'React Ninja'):
            {'slicing': 2, 'dicing': 2, 'jquery': 1}
    }

    assert aggregator.job_aggregators['skills'].rollup == {
        'Cupcake Ninja': {'slicing': 1, 'dicing': 1},
        'Regular Ninja': {'slicing': 1, 'dicing': 1},
        'React Ninja': {'slicing': 2, 'dicing': 2, 'jquery': 1}
    }


def test_geo_title_aggregator_process_postings():
    aggregator = build_basic_geo_title_aggregator()
    aggregator.process_postings([json.dumps(job) for job in SAMPLE_JOBS])
    basic_assertions(aggregator)


def test_addition_operator():
    first_aggregator = build_basic_geo_title_aggregator()
    second_aggregator = build_basic_geo_title_aggregator()

    first_aggregator.process_postings(
        [json.dumps(job) for job in SAMPLE_JOBS[0:2]]
    )
    second_aggregator.process_postings(
        [json.dumps(job) for job in SAMPLE_JOBS[2:]]
    )

    first_aggregator.merge_job_aggregators(second_aggregator.job_aggregators)
    basic_assertions(first_aggregator)


def parallelizable_aggregation(job_postings):
    aggregator = build_basic_geo_title_aggregator()
    aggregator.process_postings(job_postings)
    return aggregator.job_aggregators


def test_multiprocessing():
    pool = Pool(processes=2)
    batcher = Batch((json.dumps(job) for job in SAMPLE_JOBS), 2)
    aggregators = pool.map(
        parallelizable_aggregation,
        [list(batch) for batch in batcher]
    )
    combined_aggregator = GeoTitleAggregator(
        job_aggregators=aggregators[0],
        geo_querier=FakeCBSAQuerier()
    )
    for aggregator in aggregators[1:]:
        combined_aggregator.merge_job_aggregators(aggregator)
    basic_assertions(combined_aggregator)


def test_geo_title_aggregator_save_counts():
    job_aggregators = OrderedDict()
    job_aggregators['skills'] = SkillAggregator(
        skill_extractor=FakeExactMatchSkillExtractor(
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
    aggregator.process_postings(
        [json.dumps(job) for job in SAMPLE_JOBS + weighted_jobs]
    )

    with NamedTemporaryFile(mode='w+') as tf:
        aggregator.save_counts(tf.name)
        tf.seek(0)
        reader = csv.reader(tf)
        header_row = next(reader)
        assert header_row == [
            'cbsa_fips',
            'cbsa_name',
            'state_code',
            'title',
            'skills_1',
            'skills_2',
            'skills_3',
            'skills_total',
            'count_total'
        ]
        data_rows = [row for row in reader]
        expected = [
            [
                '123',
                'Another Metro',
                'YY',
                'Cupcake Ninja',
                'slicing',
                'dicing',
                '',
                '2',
                '2'
            ],
            [
                '456',
                'A Metro',
                'XX',
                'Regular Ninja',
                'slicing',
                'dicing',
                '',
                '2',
                '2'
            ],
            [
                '456',
                'A Metro',
                'XX',
                'React Ninja',
                'slicing',
                'dicing',
                'jquery',
                '3',
                '4'
            ],
        ]

        for expected_row in expected:
            assert expected_row in data_rows


def test_geo_title_aggregator_save_rollup():
    job_aggregators = OrderedDict()
    job_aggregators['skills'] = SkillAggregator(
        skill_extractor=FakeExactMatchSkillExtractor(
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
    aggregator.process_postings(
        [json.dumps(job) for job in SAMPLE_JOBS + weighted_jobs]
    )

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
            skill_extractor=FakeExactMatchSkillExtractor(
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
    }

    assert aggregator.job_aggregators['count'].rollup == {
        'cupcake ninja': {'total': 1},
        'regular ninja': {'total': 1},
        'react ninja': {'total': 2},
    }

    assert aggregator.job_aggregators['skills'].group_values == {
        (('123', 'Another Metro', 'YY'), 'cupcake ninja'):
            {'slicing': 1, 'dicing': 1},
        (('456', 'A Metro', 'XX'), 'regular ninja'):
            {'slicing': 1, 'dicing': 1},
        (('456', 'A Metro', 'XX'), 'react ninja'):
            {'slicing': 2, 'dicing': 2, 'jquery': 1}
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
    assert aggregate == {'23-1234.00': 1, '12-1234.00': 3}
