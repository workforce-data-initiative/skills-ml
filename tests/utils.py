import unicodecsv as csv
import json
import tempfile
import s3fs

from contextlib import contextmanager


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


def create_skills_file(skills_file_path):
    content = [
        ['', 'O*NET-SOC Code', 'Element ID', 'ONET KSA', 'Description', 'skill_uuid', 'nlp_a'],
        ['1', '11-1011.00', '2.a.1.a', 'reading comprehension', '...', '2c77c703bd66e104c78b1392c3203362', 'reading comprehension'],
        ['2', '11-1011.00', '2.a.1.b', 'active listening', '...', 'a636cb69257dcec699bce4f023a05126', 'active listening']
    ]
    s3 = s3fs.S3FileSystem()
    with s3.open(skills_file_path, 'wb') as write_stream:
        writer = csv.writer(write_stream, delimiter='\t')
        for row in content:
            writer.writerow(row)
