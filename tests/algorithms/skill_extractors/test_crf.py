import json

import boto3
from moto import mock_s3, mock_s3_deprecated

from skills_ml.algorithms.skill_extractors.crf import store_sequence_from_annotations
from skills_ml.storage import S3Store


@mock_s3_deprecated
@mock_s3
def test_store_sequence_from_annotations():
    # create a bucket that will contain the source sample

    s3 = boto3.resource('s3')
    bucket = s3.create_bucket(Bucket='test-bucket')
    # create a sample.
    # sample format is one file, one job posting per line, in common schema JSON format
    postings = {
        'ABC_91238': 'this is a job description which talks about substance abuse counseling. this is another sentence.',
        'ABC_4823943': 'job description python programming and substance abuse counseling',
    }

    bucket.put_object(
        Body='\n'.join(json.dumps({'id': key, 'description': value}) for key, value in postings.items()),
        Key='samples/test-sample'
    )

    input_annotations = [
        {
            'entity': 'Skill',
            'start_index': 44,
            'end_index': 70,
            'labeled_string': 'substance abuse counseling',
            'job_posting_id': 'ABC_91238',
            'sample_name': 'test-sample',
            'tagger_id': 'user_1',
        },
        {
            'entity': 'Skill',
            'start_index': 44,
            'end_index': 70,
            'labeled_string': 'substance abuse counseling',
            'job_posting_id': 'ABC_91238',
            'sample_name': 'test-sample',
            'tagger_id': 'user_2',
        },
        {
            'entity': 'Skill',
            'start_index': 16,
            'end_index': 33,
            'labeled_string': 'python programming',
            'job_posting_id': 'ABC_4823943',
            'sample_name': 'test-sample',
            'tagger_id': 'user_1',
        },
        {
            'entity': 'Skill',
            'start_index': 39,
            'end_index': 65,
            'labeled_string': 'substance abuse counseling',
            'job_posting_id': 'ABC_4823943',
            'sample_name': 'test-sample',
            'tagger_id': 'user_1',
        },
        {
            'entity': 'Skill',
            'start_index': 16,
            'end_index': 33,
            'labeled_string': 'python programming',
            'job_posting_id': 'ABC_4823943',
            'sample_name': 'test-sample',
            'tagger_id': 'user_2',
        },
        {
            'entity': 'Skill',
            'start_index': 49,
            'end_index': 65,
            'labeled_string': 'abuse counseling',
            'job_posting_id': 'ABC_4823943',
            'sample_name': 'test-sample',
            'tagger_id': 'user_2',
        }
    ]

    sample_base_path = 'test-bucket/samples'

    # expect one sequence per tagger and job posting combination
    expected_text = """
this O
is O
a O
job O
description O
which O
talks O
about O
substance B-SKILL
abuse I-SKILL
counseling I-SKILL
. O

this O
is O
another O
sentence O
. O

this O
is O
a O
job O
description O
which O
talks O
about O
substance B-SKILL
abuse I-SKILL
counseling I-SKILL
. O

this O
is O
another O
sentence O
. O

job O
description O
python B-SKILL
programming I-SKILL
and O
substance B-SKILL
abuse I-SKILL
counseling I-SKILL

job O
description O
python B-SKILL
programming I-SKILL
and O
substance O
abuse B-SKILL
counseling I-SKILL
"""
    output_storage = S3Store('test-bucket/sequence_file.txt')
    store_sequence_from_annotations(
        sample_base_path=sample_base_path,
        annotations=input_annotations,
        storage=output_storage,
    )
    assert output_storage.read() == expected_text
