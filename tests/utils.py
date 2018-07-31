import unicodecsv as csv
import json
import tempfile

from contextlib import contextmanager
from skills_ml.ontologies.base import CompetencyFramework, Competency
from skills_ml.algorithms.sampling import Sample
from skills_ml.storage import InMemoryStore
from skills_ml.algorithms.skill_extractors.base import CandidateSkill
import factory
import factory.fuzzy


@contextmanager
def makeNamedTemporaryCSV(content, separator=','):
    tf = tempfile.NamedTemporaryFile(delete=False)
    with open(tf.name, 'wb') as write_stream:
        writer = csv.writer(write_stream, delimiter=separator)
        for row in content:
            writer.writerow(row)

    yield tf.name

    tf.close()


class CandidateSkillFactory(factory.Factory):
    class Meta:
        model = CandidateSkill

    skill_name = factory.fuzzy.FuzzyText()
    matched_skill = factory.LazyAttribute(lambda obj: obj.skill_name)
    context = factory.LazyAttribute(lambda obj: f"{factory.fuzzy.FuzzyText()}{obj.skill_name}{factory.fuzzy.FuzzyText()}")
    confidence = factory.fuzzy.FuzzyFloat(0, 1)
    document_id = factory.fuzzy.FuzzyText()
    document_type = '@JobPosting',
    source_object = factory.LazyAttribute(lambda obj: {
        '@type': obj.document_type,
        'id': obj.document_id,
    })
    skill_extractor_name = factory.fuzzy.FuzzyText()


def job_posting_factory(**kwargs):
    with open('sample_job_listing.json') as f:
        sample_job_posting = json.load(f)
        for key, value in kwargs.items():
            if key in sample_job_posting:
                sample_job_posting[key] = value
            else:
                raise ValueError('Incorrect job posting factory override %s sent', key)
        return sample_job_posting


def sample_factory(job_postings, name='asample', storage=None):
    if not storage:
        storage = InMemoryStore()
    storage.write(
        '\n'.encode('utf-8').join(json.dumps(job_posting).encode('utf-8') for job_posting in job_postings),
        name
    )
    return Sample(storage, name)
     

def sample_framework():
    return CompetencyFramework(
        name='sample_framework',
        description='A few basic competencies',
        competencies=[
            Competency(identifier='a', name='Reading Comprehension'),
            Competency(identifier='b', name='Active Listening'),
        ]
    )
