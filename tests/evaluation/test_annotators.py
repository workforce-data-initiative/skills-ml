from skills_ml.evaluation.annotators import BratExperiment
from skills_ml.algorithms.sampling import Sample
from moto import mock_s3, mock_s3_deprecated
import boto3
import json
import s3fs
from unittest.mock import MagicMock
import pytest
import unittest


@mock_s3_deprecated
@mock_s3
def test_BratExperiment_start():
    # create a bucket that will contain both the source samples and BRAT config
    s3 = boto3.resource('s3')
    bucket = s3.create_bucket(Bucket='test-bucket')

    # create a sample.
    # sample format is one file, one job posting per line, in common schema JSON format
    bucket.put_object(
        Body='\n'.join(json.dumps({'id': i, 'description': str(i)}) for i in range(100, 200)),
        Key='samples/300_weighted'
    )

    experiment = BratExperiment(
        experiment_name='initial_skills_tag',
        brat_s3_path='test-bucket/brat'
    )
    experiment.start(
        sample=Sample(base_path='s3://test-bucket/samples', sample_name='300_weighted'),
        minimum_annotations_per_posting=2,
        max_postings_per_allocation=20,
        entities_with_shortcuts=(
            ('c', 'Competency'),
        )
    )

    # find metadata about what it created
    s3 = s3fs.S3FileSystem()

    # first assert that some shallow metadata was passed through
    assert experiment.metadata['sample_base_path'] == 's3://test-bucket/samples'
    assert experiment.metadata['sample_name'] == '300_weighted'
    assert experiment.metadata['entities_with_shortcuts'] == (('c', 'Competency'),)
    assert experiment.metadata['minimum_annotations_per_posting'] == 2
    assert experiment.metadata['max_postings_per_allocation'] == 20

    # next look at the posting texts themselves.
    # we expect them all of them to be present but split across a number of units
    units = experiment.metadata['units']
    assert len(units) == 5  # 100/20
    retrieved_descriptions = []
    for unit_name, documents in units.items():
        for posting_key, original_job_id in documents:
            # we should not expose the original posting ids
            # otherwise we don't care what the keys are but that they exist where we expect them to
            assert posting_key is not original_job_id
            with s3.open('{data_path}/.{unit_name}/{posting_key}.txt'.format(
                    data_path=experiment.data_path,
                    unit_name=unit_name,
                    posting_key=posting_key
            ), mode='rb') as f:
                posting = f.read().decode('utf-8')
                retrieved_descriptions.append(posting.strip())
            # make sure that the blank annotation file is there too
            with s3.open('{data_path}/.{unit_name}/{posting_key}.ann'.format(
                    data_path=experiment.data_path,
                    unit_name=unit_name,
                    posting_key=posting_key
            ), mode='rb') as f:
                assert len(f.read().decode('utf-8')) == 0
    # our fake descriptions were just the string values for the range numbers
    # so that's what should get written
    assert sorted(retrieved_descriptions) == sorted([str(i) for i in range(100, 200)])

    def assert_conf_contains(conf_name, expected):
        with s3.open('{path}/{conf_name}'.format(
                path=experiment.brat_config_path,
                conf_name=conf_name
        ), 'rb') as f:
            assert expected in f.read().decode('utf-8')

    assert_conf_contains('visual.conf', '[labels]\nCompetency\n')
    assert_conf_contains('annotation.conf', '[entities]\nCompetency\n')
    assert_conf_contains('kb_shortcuts.conf', 'c Competency\n')


@mock_s3
def test_BratExperiment_add_user():
    # given a valid brat experiment,
    # call add_user with a username and password
    # expect the user/pass to be added to config.py,
    # and for the user to be added to the experiment's metadata
    # and for add_allocation to be called with the user

    # setup: create a bucket for the brat config
    s3 = boto3.resource('s3')
    s3.create_bucket(Bucket='test-bucket')
    # initialize the experiment in this bucket
    experiment = BratExperiment(
        experiment_name='initial_skills_tag',
        brat_s3_path='test-bucket/brat'
    )

    # setup the add_allocation mock for later inspection
    add_allocation_mock = MagicMock()
    experiment.add_allocation = add_allocation_mock

    # add a user
    experiment.add_user('user', 'pass')

    # assert metadata
    assert 'user' in experiment.user_pw_store

    # assert that we attempted to allocate postings to them
    add_allocation_mock.assert_called_with('user')


@mock_s3
@mock_s3_deprecated
def test_BratExperiment_add_allocation():
    # given a user name
    # find the next allocation to use that the user has not annotated yet
    # create a directory with the users name
    # record in metadata the fact that the user has been allocated this

    # setup: create a bucket for the brat config
    s3 = boto3.resource('s3')
    bucket = s3.create_bucket(Bucket='test-bucket')
    bucket.put_object(
        Body='\n'.join(json.dumps({'id': i, 'description': str(i)}) for i in range(100, 200)),
        Key='samples/300_weighted'
    )

    experiment = BratExperiment(
        experiment_name='initial_skills_tag',
        brat_s3_path='test-bucket/brat'
    )
    experiment.start(
        sample=Sample(base_path='s3://test-bucket/samples', sample_name='300_weighted'),
        minimum_annotations_per_posting=2,
        max_postings_per_allocation=20,
        entities_with_shortcuts=(
            ('c', 'Competency'),
        )
    )
    # initialize the experiment in this bucket
    experiment = BratExperiment(
        experiment_name='initial_skills_tag',
        brat_s3_path='test-bucket/brat'
    )

    username = 'testuser'
    # should not be able to allocate without creating a user
    with pytest.raises(ValueError):
        experiment.add_allocation(username)

    # set up a user to allocate to
    experiment.user_pw_store[username] = 'password'
    experiment.user_pw_store.save()
    allocated_directory = experiment.add_allocation(username)

    allocations = experiment.metadata['allocations'][username]
    assert len(allocations) == 1

    s3 = s3fs.S3FileSystem()
    filenames = s3.ls(allocated_directory)
    # there should be two files for each job posting: the .txt. and the .ann
    assert len(filenames) == len(experiment.metadata['units'][allocations[0]]) * 2

    # simulate continued allocation with more users
    user_two = 'user_two'
    user_three = 'user_three'
    experiment.add_user(user_two, 'pass')
    experiment.add_user(user_three, 'pass')
    for i in range(0, 4):
        experiment.add_allocation(user_two)
        experiment.add_allocation(user_three)
    # at this point, trying to re-allocate to either user two or three
    # should fail as they have now tagged everything
    with pytest.raises(ValueError):
        experiment.add_allocation(user_two)

    # user one should still work for now
    for i in range(0, 4):
        new_directory = experiment.add_allocation(username)
        assert new_directory != allocated_directory

    # once they have seen the whole thing, no more!
    with pytest.raises(ValueError):
        experiment.add_allocation(username)


@mock_s3
@mock_s3_deprecated
def test_BratExperiment_average_observed_agreement():
    s3 = boto3.resource('s3')
    s3.create_bucket(Bucket='test-bucket')

    experiment = BratExperiment(
        experiment_name='initial_skills_tag',
        brat_s3_path='test-bucket/brat'
    )

    annotations_by_unit = {}
    annotations_by_unit['unit_1'] = {
        '0': {
            'user_1': [
                {'entity': 'Skill', 'start_index': 44, 'end_index': 70, 'labeled_string': 'substance abuse counseling'}
            ],
            'user_2': [
                {'entity': 'Skill', 'start_index': 44, 'end_index': 70, 'labeled_string': 'substance abuse counseling'}
            ]
        },
        '1': {
            'user_1': [
                {'entity': 'Skill', 'start_index': 16, 'end_index': 33, 'labeled_string': 'python programming'},
                {'entity': 'Skill', 'start_index': 39, 'end_index': 65, 'labeled_string': 'substance abuse counseling'}
            ],
            'user_2': [
                {'entity': 'Skill', 'start_index': 16, 'end_index': 33, 'labeled_string': 'python programming'},
                {'entity': 'Skill', 'start_index': 49, 'end_index': 65, 'labeled_string': 'abuse counseling'}
            ],
        }
    }
    experiment.annotations_by_unit = annotations_by_unit
    assert experiment.average_observed_agreement() == {'unit_1': {'0': 1, '1': 2/3}}


class TestLabelsWithAgreementByUnit(unittest.TestCase):
    @mock_s3
    @mock_s3_deprecated
    def test_labels_with_agreement_by_unit(self):
        s3 = boto3.resource('s3')
        bucket = s3.create_bucket(Bucket='test-bucket')

        experiment = BratExperiment(
            experiment_name='initial_skills_tag',
            brat_s3_path='test-bucket/brat'
        )

        annotations_by_unit = {}
        annotations_by_unit['unit_1'] = {
            '0': {
                'user_1': [
                    {'entity': 'Skill', 'start_index': 44, 'end_index': 70, 'labeled_string': 'substance abuse counseling'}
                ],
                'user_2': [
                    {'entity': 'Skill', 'start_index': 44, 'end_index': 70, 'labeled_string': 'substance abuse counseling'}
                ]
            },
            '1': {
                'user_1': [
                    {'entity': 'Skill', 'start_index': 16, 'end_index': 33, 'labeled_string': 'python programming'},
                    {'entity': 'Skill', 'start_index': 39, 'end_index': 65, 'labeled_string': 'substance abuse counseling'}
                ],
                'user_2': [
                    {'entity': 'Skill', 'start_index': 16, 'end_index': 33, 'labeled_string': 'python programming'},
                    {'entity': 'Skill', 'start_index': 49, 'end_index': 65, 'labeled_string': 'abuse counseling'}
                ],
            }
        }
        experiment.annotations_by_unit = annotations_by_unit
        expected = {
            'unit_1': {
                '0': [
                    {'entity': 'Skill', 'start_index': 44, 'end_index': 70, 'labeled_string': 'substance abuse counseling', 'percent_tagged': 1.0, 'number_seen': 2}
                ],
                '1': [
                    {'entity': 'Skill', 'start_index': 16, 'end_index': 33, 'labeled_string': 'python programming', 'percent_tagged': 1.0, 'number_seen': 2},
                    {'entity': 'Skill', 'start_index': 39, 'end_index': 65, 'labeled_string': 'substance abuse counseling', 'percent_tagged': 0.5, 'number_seen': 2},
                    {'entity': 'Skill', 'start_index': 49, 'end_index': 65, 'labeled_string': 'abuse counseling', 'percent_tagged': 0.5, 'number_seen': 2}
                ]
            }
        }
        self.maxDiff = None
        self.assertDictEqual(experiment.labels_with_agreement_by_unit, expected)


@mock_s3
@mock_s3_deprecated
def test_BratExperiment_labels_with_agreement():
    s3 = boto3.resource('s3')
    bucket = s3.create_bucket(Bucket='test-bucket')

    experiment = BratExperiment(
        experiment_name='initial_skills_tag',
        brat_s3_path='test-bucket/brat'
    )
    labels_by_unit = {
        'unit_1': {
            '0': [
                {'entity': 'Skill', 'start_index': 44, 'end_index': 70, 'labeled_string': 'substance abuse counseling', 'percent_tagged': 1.0, 'number_seen': 2}
            ],
            '1': [
                {'entity': 'Skill', 'start_index': 16, 'end_index': 33, 'labeled_string': 'python programming', 'percent_tagged': 1.0, 'number_seen': 2},
                {'entity': 'Skill', 'start_index': 39, 'end_index': 65, 'labeled_string': 'substance abuse counseling', 'percent_tagged': 0.5, 'number_seen': 2},
                {'entity': 'Skill', 'start_index': 49, 'end_index': 65, 'labeled_string': 'abuse counseling', 'percent_tagged': 0.5, 'number_seen': 2}
            ]
        }
    }
    experiment.labels_with_agreement_by_unit = labels_by_unit
    experiment.metadata['units'] = {
        'unit_1': [
            (0, 'ABC_91238'),
            (1, 'ABC_4823943'),
        ]
    }
    experiment.metadata['sample_name'] = 'test-sample'
    experiment.metadata.save()
    assert experiment.labels_with_agreement == [
        {'job_posting_id': 'ABC_91238', 'entity': 'Skill', 'start_index': 44, 'end_index': 70, 'labeled_string': 'substance abuse counseling', 'percent_tagged': 1.0, 'number_seen': 2, 'sample_name': 'test-sample'},
        {'job_posting_id': 'ABC_4823943', 'entity': 'Skill', 'start_index': 16, 'end_index': 33, 'labeled_string': 'python programming', 'percent_tagged': 1.0, 'number_seen': 2, 'sample_name': 'test-sample'},
        {'job_posting_id': 'ABC_4823943', 'entity': 'Skill', 'start_index': 39, 'end_index': 65, 'labeled_string': 'substance abuse counseling', 'percent_tagged': 0.5, 'number_seen': 2, 'sample_name': 'test-sample'},
        {'job_posting_id': 'ABC_4823943', 'entity': 'Skill', 'start_index': 49, 'end_index': 65, 'labeled_string': 'abuse counseling', 'percent_tagged': 0.5, 'number_seen': 2, 'sample_name': 'test-sample'}
    ]


@mock_s3
@mock_s3_deprecated
def test_BratExperiment_annotations_by_unit():
    s3 = boto3.resource('s3')
    s3.create_bucket(Bucket='test-bucket')

    experiment = BratExperiment(
        experiment_name='initial_skills_tag',
        brat_s3_path='test-bucket/brat'
    )

    # directory structure
    # 2 different units with 2 job postings each, 2 users each
    # calculate inter-rater reliability per job-posting

    job_postings = {
        'unit_1/0': 'this is a job description which talks about substance abuse counseling',
        'unit_1/1': 'job description python programming and substance abuse counseling',
        'unit_2/0': 'hello we want a person who is skilled with offender orientation and intervention techniques',
        'unit_2/1': 'development of positive social skills is important to us and also intervention techniques'
    }
    tags = {
        'user_1': {
            'unit_1/0': [
                'T1\tSkill 44 70\tsubstance abuse counseling',
            ],
            'unit_1/1': [
                'T1\tSkill 16 33\tpython programming',
                'T2\tSkill 39 65\tsubstance abuse counseling',
            ]
        },
        'user_2': {
            'unit_1/0': [
                'T1\tSkill 44 70\tsubstance abuse counseling',
            ],
            'unit_1/1': [
                'T1\tSkill 16 33\tpython programming',
                'T2\tSkill 49 65\tabuse counseling',
            ]
        },
        'user_3': {
            'unit_2/0': [
                'T1\tSkill 43 62\toffender orientation',
                'T2\tSkill 68 90\tintervention techniques',
            ],
            'unit_2/1': [
                'T1\tSkill 0 36\tdevelopment of positive social skills',
                'T2\tSkill 66 88\tintervention techniques',
            ]
        },
        'user_4': {
            'unit_2/0': [
                'T1\tSkill 43 62\toffender orientation',
                'T2\tSkill 68 90\tintervention techniques',
            ],
            'unit_2/1': []
        },
    }
    experiment.metadata['units'] = {}
    for key in job_postings.keys():
        unit_name, num = key.split('/')
        if unit_name not in experiment.metadata['units']:
            experiment.metadata['units'][unit_name] = []
        experiment.metadata['units'][unit_name].append((key, key))

    experiment.metadata['allocations'] = {}
    for user_name, annotations in tags.items():
        experiment.metadata['allocations'][user_name] = []
        for key, annotation_lines in annotations.items():
            unit_name, num = key.split('/')
            if unit_name not in experiment.metadata['allocations'][user_name]:
                experiment.metadata['allocations'][user_name].append(unit_name)

            base_path = '{}/{}'.format(experiment.user_allocations_path(user_name), key)
            with experiment.s3.open('{}.txt'.format(base_path), 'wb') as f:
                f.write(job_postings[key].encode('utf-8'))
            with experiment.s3.open('{}.ann'.format(base_path), 'wb') as f:
                f.write('\n'.join(annotation_lines).encode('utf-8'))
    experiment.metadata.save()

    assert experiment.annotations_by_unit['unit_1'] == {
        '0': {
            'user_1': [
                {'entity': 'Skill', 'start_index': 44, 'end_index': 70, 'labeled_string': 'substance abuse counseling'}
            ],
            'user_2': [
                {'entity': 'Skill', 'start_index': 44, 'end_index': 70, 'labeled_string': 'substance abuse counseling'}
            ]
        },
        '1': {
            'user_1': [
                {'entity': 'Skill', 'start_index': 16, 'end_index': 33, 'labeled_string': 'python programming'},
                {'entity': 'Skill', 'start_index': 39, 'end_index': 65, 'labeled_string': 'substance abuse counseling'}
            ],
            'user_2': [
                {'entity': 'Skill', 'start_index': 16, 'end_index': 33, 'labeled_string': 'python programming'},
                {'entity': 'Skill', 'start_index': 49, 'end_index': 65, 'labeled_string': 'abuse counseling'}
            ],
        }
    }

