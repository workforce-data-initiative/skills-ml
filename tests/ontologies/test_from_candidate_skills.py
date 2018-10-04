from tests.utils import CandidateSkillFactory
from skills_ml.ontologies.from_candidate_skills import ontology_from_candidate_skills


def test_ontology_from_candidate_skills():
    candidate_skills = [CandidateSkillFactory(skill_name=f'skill_{i}') for i in range(0, 25)]
    ontology = ontology_from_candidate_skills(candidate_skills, skill_extractor_name='tester')
    assert ontology.name == 'candidate_skill_tester'
    assert ontology.competency_framework.name == 'candidate_skill_tester'
    assert 'tester' in ontology.competency_framework.description
    assert len(ontology.competencies) == 25
    assert len(ontology.occupations) == 1


def test_ontology_from_candidate_skills_occupations():
    candidate_skills = [CandidateSkillFactory(
        skill_name=f'skill_{i}',
        source_object={'onet_soc_code': f'11-101{i%5}.00'}
    ) for i in range(0, 25)]
    ontology = ontology_from_candidate_skills(candidate_skills)
    assert len(ontology.competencies) == 25
    assert len(ontology.occupations) == 5
