from collections import Counter

from . import sample_ontology, sample_job_posting

from skills_ml.algorithms.skill_extractors import SocScopedExactMatchSkillExtractor
from skills_ml.ontologies.base import CompetencyOntology, Competency, Occupation, CompetencyOccupationEdge


def test_occupation_scoped_freetext_skill_extractor():
    ontology = CompetencyOntology(
        competency_name='Sample Framework',
        competency_description='A few basic competencies',
        edges=[
            CompetencyOccupationEdge(
                competency=Competency(identifier='2.a.1.a', name='Reading Comprehension'),
                occupation=Occupation(identifier='11-1011.00')
            ),
            CompetencyOccupationEdge(
                competency=Competency(identifier='2.a.1.b', name='Active Listening'),
                occupation=Occupation(identifier='11-1011.00')
            ),
        ]
    )
    extractor = SocScopedExactMatchSkillExtractor(ontology)
    documents = [
        {
            'id': '1234',
            '@type': 'JobPosting',
            'onet_soc_code': '11-1011.00',
            'description': 'this is a job that needs active listening', 
            'expected_value': Counter({'active listening': 1})
        },
        {
            'id': '2234',
            '@type': 'JobPosting',
            'onet_soc_code': '11-1011.00',
            'description': 'this is a reading comprehension job',
            'expected_value': Counter({'reading comprehension': 1})
        },
        {
            'id': '3234',
            '@type': 'JobPosting',
            'onet_soc_code': '11-1011.00',
            'description': 'this is an active and reading listening job', 
            'expected_value': Counter(),
        },
        {
            'id': '4234',
            '@type': 'JobPosting',
            'onet_soc_code': '11-1011.00',
            'description': 'this is a reading comprehension and active listening job', 
            'expected_value': Counter({'active listening': 1, 'reading comprehension': 1})
        },
        {
            'id': '5234',
            '@type': 'JobPosting',
            'onet_soc_code': '11-1021.00',
            'description': 'this is a job that needs active listening', 
            'expected_value': Counter()
        },
        {
            'id': '6234',
            '@type': 'JobPosting',
            'onet_soc_code': '11-1021.00',
            'description': 'this is a reading comprehension job',
            'expected_value': Counter()
        },
        {
            'id': '7234',
            '@type': 'JobPosting',
            'onet_soc_code': '11-1021.00',
            'description': 'this is an active and reading listening job', 
            'expected_value': Counter(),
        },
        {
            'id': '8234',
            '@type': 'JobPosting',
            'onet_soc_code': '11-1021.00',
            'description': 'this is a reading comprehension and active listening job', 
            'expected_value': Counter()
        },
        {
            'id': '9234',
            '@type': 'JobPosting',
            'onet_soc_code': None,
            'description': 'this is a job that needs active listening', 
            'expected_value': Counter()
        },
        {
            'id': '1334',
            '@type': 'JobPosting',
            'onet_soc_code': None,
            'description': 'this is a reading comprehension job',
            'expected_value': Counter()
        },
        {
            'id': '1434',
            '@type': 'JobPosting',
            'onet_soc_code': None,
            'description': 'this is an active and reading listening job', 
            'expected_value': Counter(),
        },
        {
            'id': '1534',
            '@type': 'JobPosting',
            'onet_soc_code': None,
            'description': 'this is a reading comprehension and active listening job', 
            'expected_value': Counter()
        },
    ]
    for document in documents:
        assert extractor.document_skill_counts(document) == document['expected_value']



def test_occupational_scoped_skill_extractor_candidate_skills():
    extractor = SocScopedExactMatchSkillExtractor(sample_ontology())
    candidate_skills = sorted(
        extractor.candidate_skills(sample_job_posting()),
        key=lambda cs: cs.skill_name
    )

    assert candidate_skills[0].skill_name == 'organization'
    assert candidate_skills[0].context == 'Organization, Cleanliness, Trainability, team player, good communication skillz, Motivation, a Sense of Responsibility and Pride in your Performance'
    assert candidate_skills[0].confidence == 100
