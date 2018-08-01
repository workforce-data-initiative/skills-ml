import unicodecsv as csv
import json
import tempfile

from contextlib import contextmanager
from skills_ml.ontologies.base import CompetencyFramework, Competency


@contextmanager
def makeNamedTemporaryCSV(content, separator=','):
    tf = tempfile.NamedTemporaryFile(delete=False)
    with open(tf.name, 'wb') as write_stream:
        writer = csv.writer(write_stream, delimiter=separator)
        for row in content:
            writer.writerow(row)

    yield tf.name

    tf.close()


def job_posting_factory(**kwargs):
    with open('sample_job_listing.json') as f:
        sample_job_posting = json.load(f)
        for key, value in kwargs.items():
            if key in sample_job_posting:
                sample_job_posting[key] = value
            else:
                raise ValueError('Incorrect job posting factory override %s sent', key)
        return sample_job_posting


def sample_framework():
    return CompetencyFramework(
        name='sample_framework',
        description='A few basic competencies',
        competencies=[
            Competency(identifier='a', name='Reading Comprehension'),
            Competency(identifier='b', name='Active Listening'),
        ]
    )
