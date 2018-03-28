from collections import Counter

from tests import utils
from . import sample_skills, sample_job_posting

from skills_ml.algorithms.skill_extractors import SocScopedExactMatchSkillExtractor


def test_occupation_scoped_freetext_skill_extractor():
    content = [
        ['', 'O*NET-SOC Code', 'Element ID', 'ONET KSA', 'Description', 'skill_uuid', 'nlp_a'],
        ['1', '11-1011.00', '2.a.1.a', 'reading comprehension', '...', '2c77c703bd66e104c78b1392c3203362', 'reading comprehension'],
        ['2', '11-1011.00', '2.a.1.b', 'active listening', '...', 'a636cb69257dcec699bce4f023a05126', 'active listening']
    ]
    with utils.makeNamedTemporaryCSV(content, '\t') as skills_filename:
        extractor = SocScopedExactMatchSkillExtractor(skill_lookup_path=skills_filename)
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



def test_occupational_scoped_skill_extractor_candidate_skills():
    with utils.makeNamedTemporaryCSV(sample_skills(), '\t') as skills_filename:
        extractor = SocScopedExactMatchSkillExtractor(skill_lookup_path=skills_filename)
        candidate_skills = sorted(
            extractor.candidate_skills(sample_job_posting()),
            key=lambda cs: cs.skill_name
        )

        assert candidate_skills[0].skill_name == 'organization'
        assert candidate_skills[0].context == 'Organization, Cleanliness, Trainability, team player, good communication skillz, Motivation, a Sense of Responsibility and Pride in your Performance'
        assert candidate_skills[0].confidence == 100
