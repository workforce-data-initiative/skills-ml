import json
import pytest

from skills_ml.job_postings import JobPosting
from skills_ml.ontologies.base import CompetencyFramework, Competency, Occupation, CompetencyOntology, CompetencyOccupationEdge


@pytest.fixture
def sample_job_posting():
    return {
        "id": "TEST_12345",
        "description": "The Hall Line Cook will maintain and prepare hot and cold foods for the\nrestaurant according to Chefs specifications and for catered events as\nrequired. One-two years cooking experience in a professional kitchen\nenvironment is desired, but willing to train someone with a positive attitude,\ndesire to learn and passion for food and service. Qualified candidates will\nhave the ability to follow directions, as well as being self directed.\nOrganization, Cleanliness, Trainability, team player, good communication skillz, Motivation, a Sense of Responsibility and Pride in your Performance\nare ESSENTIAL.",
        "onet_soc_code": '11-1012.00',
    }


def sample_framework():
    return CompetencyFramework(
        name='Sample Framework',
        description='A few basic competencies',
        competencies=[
            Competency(identifier='a', name='Organization'),
            Competency(identifier='b', name='Communication Skills'),
            Competency(identifier='c', name='Cooking')
        ]
    )

def sample_ontology():
    return CompetencyOntology(
        competency_name='Sample Framework',
        competency_description='A few basic competencies',
        edges=[
            CompetencyOccupationEdge(
                competency=Competency(identifier='a', name='Organization'),
                occupation=Occupation(identifier='11-1011.00')
            ),
            CompetencyOccupationEdge(
                competency=Competency(identifier='a', name='Organization'),
                occupation=Occupation(identifier='11-1012.00')
            ),
            CompetencyOccupationEdge(
                competency=Competency(identifier='b', name='Communication Skills'),
                occupation=Occupation(identifier='11-1011.00')
            ),
            CompetencyOccupationEdge(
                competency=Competency(identifier='c', name='Cooking'),
                occupation=Occupation(identifier='11-1011.00')
            ),
        ]
    )
