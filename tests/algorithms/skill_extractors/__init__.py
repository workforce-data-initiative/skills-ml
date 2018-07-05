import json
import pytest

from skills_ml.job_postings import JobPosting


@pytest.fixture
def sample_job_posting():
    return {
        "id": "TEST_12345",
        "description": "The Hall Line Cook will maintain and prepare hot and cold foods for the\nrestaurant according to Chefs specifications and for catered events as\nrequired. One-two years cooking experience in a professional kitchen\nenvironment is desired, but willing to train someone with a positive attitude,\ndesire to learn and passion for food and service. Qualified candidates will\nhave the ability to follow directions, as well as being self directed.\nOrganization, Cleanliness, Trainability, team player, good communication skillz, Motivation, a Sense of Responsibility and Pride in your Performance\nare ESSENTIAL.",
        "onet_soc_code": '11-1012.00',
    }


@pytest.fixture
def sample_skills():
    return [
        ['', 'O*NET-SOC Code', 'Element ID', 'ONET KSA', 'Description', 'skill_uuid', 'nlp_a'],
        ['1', '11-1011.00', '2.a.1.a', 'organization', '...', '...', 'organization'],
        ['2', '11-1011.00', '2.a.1.b', 'communication skills', '...', '...', 'communication skills'],
        ['3', '11-1011.00', '2.a.1.b', 'cooking', '...', '...', 'cooking'],
        ['4', '11-1012.00', '2.a.1.a', 'organization', '...', '...', 'organization'],
    ]
