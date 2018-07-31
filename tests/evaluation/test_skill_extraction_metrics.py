from skills_ml.evaluation.skill_extraction_metrics import (
    OntologyCompetencyRecall,
    OntologyOccupationRecall,
    MedianSkillsPerDocument,
    PercentageNoSkillDocuments,
    TotalVocabularySize,
    TotalOccurrences
)
from tests.algorithms.skill_extractors import sample_ontology
from tests.utils import CandidateSkillFactory


def test_TotalOccurrences():
    candidate_skills = CandidateSkillFactory.create_batch(50)
    assert TotalOccurrences().eval(candidate_skills, 100) == 50


def test_TotalVocabularySize():
    candidate_skills = [
        CandidateSkillFactory(skill_name='skill_' + str(i % 10))
        for i in range(0, 100)
    ]
    assert TotalVocabularySize().eval(candidate_skills, 100) == 10


def test_MedianSkillsPerDocument():
    # candidate skills are created:
    # 10 for docid '0', 10 for docid '1', 5 for docid '2'.
    # at this point, the median would be 10
    candidate_skills = [
        CandidateSkillFactory(document_id=str(int(i / 10)))
        for i in range(0, 25)
    ]
    # but we indicate that the sample had 5 total documents
    # the count array should look like [10, 10, 5, 0, 0]
    # and the median should be 5
    assert MedianSkillsPerDocument()\
        .eval(candidate_skills, sample_len=5) == 5


def test_PercentageNoSkillDocument():
    candidate_skills = [
        CandidateSkillFactory(document_id=str(i))
        for i in range(0, 25)
    ]
    # 25 documents with skills, 5 without. should be 5/30
    assert PercentageNoSkillDocuments()\
        .eval(candidate_skills, 30) == 5/30


def test_OntologyCompetencyRecall():
    ontology = sample_ontology()
    metric = OntologyCompetencyRecall(ontology)
    candidate_skills = CandidateSkillFactory.create_batch(
        50,
        skill_name=list(ontology.competencies)[0].name.lower()
    )
    assert metric.eval(candidate_skills, 50) ==\
        1/len(ontology.competencies)


def test_OntologyOccupationRecall():
    ontology = sample_ontology()
    metric = OntologyOccupationRecall(ontology)
    candidate_skills = [
        CandidateSkillFactory(
            document_id=str(i),
            source_object={'occupationalCategory': list(ontology.occupations)[0].identifier}
        )
        for i in range(0, 25)
    ]
    assert metric.eval(candidate_skills, 50) == 1/len(ontology.occupations)
