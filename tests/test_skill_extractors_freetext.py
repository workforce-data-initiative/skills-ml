from collections import Counter
import json

from tests import utils
import pytest

from skills_ml.algorithms.skill_extractors import JobPosting
from skills_ml.algorithms.skill_extractors.freetext import\
    ExactMatchSkillExtractor,\
    OccupationScopedSkillExtractor,\
    FuzzyMatchSkillExtractor


def test_exactmatch_skill_extractor():
    content = [
        ['', 'O*NET-SOC Code', 'Element ID', 'ONET KSA', 'Description', 'skill_uuid', 'nlp_a'],
        ['1', '11-1011.00', '2.a.1.a', 'reading comprehension', '...', '2c77c703bd66e104c78b1392c3203362', 'reading comprehension'],
        ['2', '11-1011.00', '2.a.1.b', 'active listening', '...', 'a636cb69257dcec699bce4f023a05126', 'active listening']
    ]
    with utils.makeNamedTemporaryCSV(content, '\t') as skills_filename:
        extractor = ExactMatchSkillExtractor(skill_lookup_path=skills_filename)
        result = [extractor.document_skill_counts(doc) for doc in [
            'this is a job that needs active listening',
            'this is a reading comprehension job',
            'this is an active and reading listening job',
            'this is a reading comprehension and active listening job',
        ]]

        assert result == [
            Counter({'active listening': 1}),
            Counter({'reading comprehension': 1}),
            Counter(),
            Counter({'active listening': 1, 'reading comprehension': 1})
        ]


def test_occupation_scoped_freetext_skill_extractor():
    content = [
        ['', 'O*NET-SOC Code', 'Element ID', 'ONET KSA', 'Description', 'skill_uuid', 'nlp_a'],
        ['1', '11-1011.00', '2.a.1.a', 'reading comprehension', '...', '2c77c703bd66e104c78b1392c3203362', 'reading comprehension'],
        ['2', '11-1011.00', '2.a.1.b', 'active listening', '...', 'a636cb69257dcec699bce4f023a05126', 'active listening']
    ]
    with utils.makeNamedTemporaryCSV(content, '\t') as skills_filename:
        extractor = OccupationScopedSkillExtractor(skill_lookup_path=skills_filename)
        documents = [
            {
                'soc_code': '11-1011.00',
                'document': 'this is a job that needs active listening', 
                'expected_value': Counter({'active listening': 1})
            },
            {
                'soc_code': '11-1011.00',
                'document': 'this is a reading comprehension job',
                'expected_value': Counter({'reading comprehension': 1})
            },
            {
                'soc_code': '11-1011.00',
                'document': 'this is an active and reading listening job', 
                'expected_value': Counter(),
            },
            {
                'soc_code': '11-1011.00',
                'document': 'this is a reading comprehension and active listening job', 
                'expected_value': Counter({'active listening': 1, 'reading comprehension': 1})
            },
            {
                'soc_code': '11-1021.00',
                'document': 'this is a job that needs active listening', 
                'expected_value': Counter()
            },
            {
                'soc_code': '11-1021.00',
                'document': 'this is a reading comprehension job',
                'expected_value': Counter()
            },
            {
                'soc_code': '11-1021.00',
                'document': 'this is an active and reading listening job', 
                'expected_value': Counter(),
            },
            {
                'soc_code': '11-1021.00',
                'document': 'this is a reading comprehension and active listening job', 
                'expected_value': Counter()
            },
            {
                'soc_code': None,
                'document': 'this is a job that needs active listening', 
                'expected_value': Counter()
            },
            {
                'soc_code': None,
                'document': 'this is a reading comprehension job',
                'expected_value': Counter()
            },
            {
                'soc_code': None,
                'document': 'this is an active and reading listening job', 
                'expected_value': Counter(),
            },
            {
                'soc_code': None,
                'document': 'this is a reading comprehension and active listening job', 
                'expected_value': Counter()
            },
        ]
        for document in documents:
            assert extractor.document_skill_counts(
                soc_code=document['soc_code'],
                document=document['document']
            ) == document['expected_value']


@pytest.fixture
def sample_job_posting():
    return JobPosting(json.dumps({
        "id": "TEST_12345",
        "description": "The Hall Line Cook will maintain and prepare hot and cold foods for the\nrestaurant according to Chefs specifications and for catered events as\nrequired. One-two years cooking experience in a professional kitchen\nenvironment is desired, but willing to train someone with a positive attitude,\ndesire to learn and passion for food and service. Qualified candidates will\nhave the ability to follow directions, as well as being self directed.\nOrganization, Cleanliness, Trainability, team player, good communication skillz, Motivation, a Sense of Responsibility and Pride in your Performance\nare ESSENTIAL.",
        "onet_soc_code": '11-1012.00',
    }).encode('utf-8'))

@pytest.fixture
def sample_skills():
    return [
        ['', 'O*NET-SOC Code', 'Element ID', 'ONET KSA', 'Description', 'skill_uuid', 'nlp_a'],
        ['1', '11-1011.00', '2.a.1.a', 'organization', '...', '...', 'organization'],
        ['2', '11-1011.00', '2.a.1.b', 'communication skills', '...', '...', 'communication skills'],
        ['3', '11-1011.00', '2.a.1.b', 'cooking', '...', '...', 'cooking'],
        ['4', '11-1012.00', '2.a.1.a', 'organization', '...', '...', 'organization'],
    ]

def test_exactmatch_skill_extractor_candidate_skills():
    with utils.makeNamedTemporaryCSV(sample_skills(), '\t') as skills_filename:
        extractor = ExactMatchSkillExtractor(skill_lookup_path=skills_filename)
        candidate_skills = sorted(
            extractor.candidate_skills(sample_job_posting()),
            key=lambda cs: cs.skill_name
        )

        assert candidate_skills[0].skill_name == 'cooking'
        assert candidate_skills[0].context == 'One-two years cooking experience in a professional kitchen'
        assert candidate_skills[0].confidence == 100

        assert candidate_skills[1].skill_name == 'organization'
        assert candidate_skills[1].context == 'Organization, Cleanliness, Trainability, team player, good communication skillz, Motivation, a Sense of Responsibility and Pride in your Performance'
        assert candidate_skills[1].confidence == 100

def test_occupational_scoped_skill_extractor_candidate_skills():
    with utils.makeNamedTemporaryCSV(sample_skills(), '\t') as skills_filename:
        extractor = OccupationScopedSkillExtractor(skill_lookup_path=skills_filename)
        candidate_skills = sorted(
            extractor.candidate_skills(sample_job_posting()),
            key=lambda cs: cs.skill_name
        )

        assert candidate_skills[0].skill_name == 'organization'
        assert candidate_skills[0].context == 'Organization, Cleanliness, Trainability, team player, good communication skillz, Motivation, a Sense of Responsibility and Pride in your Performance'
        assert candidate_skills[0].confidence == 100


def test_fuzzymatch_skill_extractor_candidate_skills():
    with utils.makeNamedTemporaryCSV(sample_skills(), '\t') as skills_filename:
        extractor = FuzzyMatchSkillExtractor(skill_lookup_path=skills_filename)
        candidate_skills = sorted(
            extractor.candidate_skills(sample_job_posting()),
            key=lambda cs: cs.skill_name
        )

        assert candidate_skills[0].skill_name == 'communication skills'
        assert candidate_skills[0].context == 'Organization, Cleanliness, Trainability, team player, good communication skillz, Motivation, a Sense of Responsibility and Pride in your Performance'
        assert candidate_skills[0].confidence == 95

        assert candidate_skills[1].skill_name == 'cooking'
        assert candidate_skills[1].context == 'One-two years cooking experience in a professional kitchen'
        assert candidate_skills[1].confidence == 100

        assert candidate_skills[2].skill_name == 'organization'
        assert candidate_skills[2].context == 'Organization, Cleanliness, Trainability, team player, good communication skillz, Motivation, a Sense of Responsibility and Pride in your Performance'
        assert candidate_skills[2].confidence == 100
