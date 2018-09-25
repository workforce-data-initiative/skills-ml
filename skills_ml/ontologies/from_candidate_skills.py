from collections import defaultdict

from skills_ml.ontologies.base import Competency, Occupation, CompetencyOntology
from skills_ml.algorithms.skill_extractors.base import CandidateSkillYielder
from skills_ml.job_postings.common_schema import get_onet_occupation


def ontology_from_candidate_skills(candidate_skills: CandidateSkillYielder, skill_extractor_name: str='unknown') -> CompetencyOntology:
    """Create an ontology from a list of candidate skills

    Simply associate each candidate skill with its ONET occupation.

    Args:
        candidate_skills (iterable of algorithms.skill_extractors.base.CandidateSkill objects)

    Returns: (skills_ml.ontologies.base.CompetencyOntology)
    """
    ontology = CompetencyOntology(
        name=f'candidate_skill_{skill_extractor_name}',
        competency_name=f'candidate_skill_{skill_extractor_name}',
        competency_description=f'Constructed from CandidateSkill objects produced by the {skill_extractor_name} skill extractor'
    )
    competencies_by_document_id = defaultdict(set)
    for candidate_skill in candidate_skills:
        competency = Competency(
            identifier=candidate_skill.skill_name.lower(),
            name=candidate_skill.skill_name
        )
        if competency not in competencies_by_document_id[candidate_skill.document_id]:
            competencies_by_document_id[candidate_skill.document_id].add(competency)
        if competency not in ontology.competencies:
            ontology.add_competency(competency)
        occupation_code = get_onet_occupation(candidate_skill.source_object)
        occupation = Occupation(identifier=occupation_code)
        if occupation not in ontology.occupations:
            ontology.add_occupation(occupation)
        ontology.add_edge(occupation=occupation, competency=competency)

    return ontology
