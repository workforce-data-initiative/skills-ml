from collections import Counter

from tests import utils
from . import sample_framework, sample_job_posting

from skills_ml.algorithms.skill_extractors import ExactMatchSkillExtractor
from skills_ml.ontologies.base import CompetencyFramework, Competency


def test_exactmatch_skill_extractor():
    competency_framework = CompetencyFramework(
        name='test_competencies',
        description='Test competencies',
        competencies=[
            Competency(identifier='2.a.1.a', name='Reading Comprehension'),
            Competency(identifier='2.a.1.b', name='Active Listening'),
        ]
    )
    extractor = ExactMatchSkillExtractor(competency_framework)
    assert competency_framework.name in extractor.name
    assert competency_framework.description in extractor.description

    result = [extractor.document_skill_counts({'id': '1234', '@type': 'JobPosting', 'description': doc}) for doc in [
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
    extractor = ExactMatchSkillExtractor(sample_framework())
    candidate_skills = sorted(
        extractor.candidate_skills(sample_job_posting()),
        key=lambda cs: cs.skill_name
    )

    assert candidate_skills[0].skill_name == 'cooking'
    assert candidate_skills[0].matched_skill_identifier == 'c'
    assert candidate_skills[0].context == 'One-two years cooking experience in a professional kitchen'
    assert candidate_skills[0].confidence == 100
    assert candidate_skills[0].start_index == 164

    assert candidate_skills[1].skill_name == 'organization'
    assert candidate_skills[1].matched_skill_identifier == 'a'
    assert candidate_skills[1].context == 'Organization, Cleanliness, Trainability, team player, good communication skillz, Motivation, a Sense of Responsibility and Pride in your Performance'
    assert candidate_skills[1].confidence == 100
    assert candidate_skills[1].start_index == 430
