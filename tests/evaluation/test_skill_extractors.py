from skills_ml.algorithms.skill_extractors import SkillEndingPatternExtractor
from skills_ml.algorithms.skill_extractors.base import CandidateSkill
from tests.utils import job_posting_factory, sample_factory
from skills_ml.evaluation.skill_extractors import candidate_skills_from_sample, metrics_for_candidate_skills
from skills_ml.evaluation.skill_extraction_metrics import TotalOccurrences, TotalVocabularySize
from skills_ml.storage import InMemoryStore
from tests.utils import CandidateSkillFactory
import json


def standard_sample():
    job_postings = [
        job_posting_factory(description='this is a job that requires communication skills')
        for _ in range(0, 5)
    ]
    sample = sample_factory(job_postings, name='mysample')
    return sample

def test_candidate_skills_from_sample_nostore():
    candidate_skills = candidate_skills_from_sample(
        standard_sample(),
        SkillEndingPatternExtractor(only_bulleted_lines=False)
    )
    assert len(candidate_skills) == 5
    for candidate_skill in candidate_skills:
        assert candidate_skill.document_type == 'JobPosting'


def test_candidate_skills_from_sample_withstore():
    storage = InMemoryStore()
    candidate_skills = candidate_skills_from_sample(
        standard_sample(),
        SkillEndingPatternExtractor(only_bulleted_lines=False),
        output_storage=storage
    )
    stored_objects = json.loads(storage.load('mysample/candidate_skills.json'))
    converted_to_candidate_skills = [CandidateSkill(**obj) for obj in stored_objects]
    assert converted_to_candidate_skills == candidate_skills


def test_metrics_for_candidate_skills_nostore():
    candidate_skills = CandidateSkillFactory.create_batch(50)
    metrics = metrics_for_candidate_skills(
        candidate_skills=candidate_skills,
        sample=standard_sample(),
        metrics=[TotalOccurrences(), TotalVocabularySize()]
    )
    assert len(metrics) == 2
    assert TotalOccurrences.name in metrics
    assert TotalVocabularySize.name in metrics


def test_metrics_for_candidate_skills_withstore():
    storage = InMemoryStore()
    candidate_skills = CandidateSkillFactory.create_batch(50)
    metrics = metrics_for_candidate_skills(
        candidate_skills=candidate_skills,
        sample=standard_sample(),
        metrics=[TotalOccurrences(), TotalVocabularySize()],
        output_storage=storage
    )

    assert json.loads(storage.load('mysample/metrics.json')) == metrics


def test_integrated_with_storage():
    storage = InMemoryStore()
    sample = standard_sample()
    candidate_skills_from_sample(
        sample,
        SkillEndingPatternExtractor(only_bulleted_lines=False),
        output_storage=storage
    )
    metrics_for_candidate_skills(
        sample=sample,
        metrics=[TotalOccurrences(), TotalVocabularySize()],
        input_storage=storage,
        output_storage=storage
    )

    assert len(json.loads(storage.load('mysample/metrics.json'))) == 2
