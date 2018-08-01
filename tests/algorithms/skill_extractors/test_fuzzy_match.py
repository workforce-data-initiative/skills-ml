from tests.algorithms.skill_extractors import sample_framework, sample_job_posting

from skills_ml.algorithms.skill_extractors import FuzzyMatchSkillExtractor


def test_fuzzymatch_skill_extractor_candidate_skills():
    extractor = FuzzyMatchSkillExtractor(sample_framework())
    candidate_skills = sorted(
        extractor.candidate_skills(sample_job_posting()),
        key=lambda cs: cs.skill_name
    )

    assert candidate_skills[0].skill_name == 'communication skillz'
    assert candidate_skills[0].matched_skill == 'communication skills'
    assert candidate_skills[0].context == 'Organization, Cleanliness, Trainability, team player, good communication skillz, Motivation, a Sense of Responsibility and Pride in your Performance'
    assert candidate_skills[0].confidence == 95

    assert candidate_skills[1].skill_name == 'cooking'
    assert candidate_skills[1].context == 'One-two years cooking experience in a professional kitchen'
    assert candidate_skills[1].confidence == 100

    assert candidate_skills[2].skill_name == 'organization'
    assert candidate_skills[2].context == 'Organization, Cleanliness, Trainability, team player, good communication skillz, Motivation, a Sense of Responsibility and Pride in your Performance'
    assert candidate_skills[2].confidence == 100
