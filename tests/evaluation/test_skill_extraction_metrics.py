from skills_ml.evaluation.skill_extraction_metrics import (
    OntologyCompetencyRecall,
    OntologyOccupationRecall,
    MedianSkillsPerDocument,
    SkillsPerDocumentHistogram,
    PercentageNoSkillDocuments,
    TotalVocabularySize,
    TotalOccurrences,
    EvaluationSetPrecision,
    EvaluationSetRecall
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


def test_SkillsPerDocumentHistogram():
    # 11 documents, each 1-9 having that # of candidate skills
    # we tell the metric that there were 13 total documents, meaning that the extra two had 0
    candidate_skills = []
    for document_id in range(1, 12):
        for cs in range(0, document_id):
            candidate_skills.append(CandidateSkillFactory(document_id=str(document_id)))

    assert SkillsPerDocumentHistogram(5).eval(candidate_skills, 13) == [4, 2, 2, 2, 3]


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
        matched_skill_identifier=list(ontology.competencies)[0].identifier.lower()
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


def test_EvaluationSetPrecision():
    # create a set of gold standard skills that are a subset of the ones being evaluated
    gold_standard_candidate_skills = [
        CandidateSkillFactory(document_id=str(i), skill_name=str(i), start_index=i)
        for i in range(0, 25)
    ]
    candidate_skills = [
        CandidateSkillFactory(document_id=str(i), skill_name=str(i), start_index=i)
        for i in range(0, 40)
    ]
    # in both strict and non-strict mode, all should match
    strict = EvaluationSetPrecision(gold_standard_candidate_skills, 'test', strict=True)
    assert strict.name == 'test_evaluation_set_precision_strict'
    assert strict.eval(candidate_skills, 40) == 25/40
    nonstrict = EvaluationSetPrecision(gold_standard_candidate_skills, 'test', strict=False)
    assert nonstrict.name == 'test_evaluation_set_precision_nonstrict'
    assert nonstrict.eval(candidate_skills, 40) == 25/40

    # now create candidate skills to evaluate that match for everything but the index is off
    # strict mode should reject all matches. non-strict will have the same result
    candidate_skills = [
        CandidateSkillFactory(document_id=str(i), skill_name=str(i), start_index=i+1)
        for i in range(0, 40)
    ]
    assert strict.eval(candidate_skills, 40) == 0/40
    assert nonstrict.eval(candidate_skills, 40) == 25/40


def test_EvaluationSetRecall():
    # create a set of gold standard skills that are a superset of the ones being evaluated
    gold_standard_candidate_skills = [
        CandidateSkillFactory(document_id=str(i), skill_name=str(i), start_index=i)
        for i in range(0, 40)
    ]
    candidate_skills = [
        CandidateSkillFactory(document_id=str(i), skill_name=str(i), start_index=i)
        for i in range(0, 25)
    ]

    # both strict and non-strict should find all of the intersection skills to match
    strict = EvaluationSetRecall(gold_standard_candidate_skills, 'test')
    assert strict.name == 'test_evaluation_set_recall_strict'
    assert strict.eval(candidate_skills, 40) == 25/40
    nonstrict = EvaluationSetRecall(gold_standard_candidate_skills, 'test', strict=False)
    assert nonstrict.name == 'test_evaluation_set_recall_nonstrict'
    assert nonstrict.eval(candidate_skills, 40) == 25/40

    # now create candidate skills to evaluate that match for everything but the index is off
    # strict mode should reject all matches. non-strict will have the same result
    candidate_skills = [
        CandidateSkillFactory(document_id=str(i), skill_name=str(i), start_index=i+1)
        for i in range(0, 25)
    ]
    assert strict.eval(candidate_skills, 40) == 0
    assert nonstrict.eval(candidate_skills, 40) == 25/40
