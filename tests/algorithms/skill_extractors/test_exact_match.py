from collections import Counter

from tests import utils
from . import sample_skills, sample_job_posting

from skills_ml.algorithms.skill_extractors import ExactMatchSkillExtractor


def test_exactmatch_skill_extractor():
    content = [
        ['', 'O*NET-SOC Code', 'Element ID', 'ONET KSA', 'Description', 'skill_uuid', 'nlp_a'],
        ['1', '11-1011.00', '2.a.1.a', 'reading comprehension', '...', '2c77c703bd66e104c78b1392c3203362', 'reading comprehension'],
        ['2', '11-1011.00', '2.a.1.b', 'active listening', '...', 'a636cb69257dcec699bce4f023a05126', 'active listening']
    ]
    with utils.makeNamedTemporaryCSV(content, '\t') as skills_filename:
        extractor = ExactMatchSkillExtractor(skill_lookup_path=skills_filename)
        result = [extractor.document_skill_counts({'description': doc}) for doc in [
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
