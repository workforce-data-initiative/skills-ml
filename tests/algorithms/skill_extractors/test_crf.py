import json

import boto3
from moto import mock_s3, mock_s3_deprecated

from skills_ml.algorithms.skill_extractors.crf import CrfTransformer
from skills_ml.storage import S3Store


@mock_s3_deprecated
@mock_s3
def test_CrfTransformerAnnotations():
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

    sample_base_path = 's3://test-bucket/samples'

    # expect one sequence per tagger and job posting combination
    expected_files = {
        'ABC_4823943-user_2': \
"""job O
description O
python B-SKILL
programming I-SKILL
and O
substance O
abuse B-SKILL
counseling I-SKILL""",
        'ABC_4823943-user_1': \
"""job O
description O
python B-SKILL
programming I-SKILL
and O
substance B-SKILL
abuse I-SKILL
counseling I-SKILL""",
        'ABC_91238-user_2': \
"""this O
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
. O""",
        'ABC_91238-user_1': \
"""this O
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
. O"""
    }

    storage_engine = S3Store('test-bucket')
    transformer = CrfTransformer(
        sample_base_path=sample_base_path,
        storage_engine=storage_engine,
    )
    transformer.transform_and_save_annotations(
        annotations=input_annotations
    )
    for filename, expected_text in expected_files.items():
        assert storage_engine.load(filename + '.txt').decode('utf-8') == expected_text
