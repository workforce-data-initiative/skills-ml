import moto
import boto

from skills_ml.job_postings.aggregate.field_values import FieldValueCounter


def test_field_value_counter():
    counter = FieldValueCounter(quarter='2014Q1', field_values=['jobtitle', 'jobdesc'])
    counter.track(
        input_document={'jobtitle': 'test', 'jobdesc': 'test'},
    )
    counter.track(
        input_document={'jobtitle': 'test', 'jobdesc': '', 'extra': 'test'},
    )
    assert counter.accumulator['jobtitle']['test'] == 2
    assert counter.accumulator['jobdesc']['test'] == 1
    assert counter.accumulator['jobdesc'][''] == 1

    with moto.mock_s3_deprecated():
        s3_conn = boto.connect_s3()
        s3_conn.create_bucket('test-bucket')
        counter.save(s3_conn, 'test-bucket/stats')

        key = s3_conn.get_bucket('test-bucket')\
            .get_key('stats/field_values/2014Q1/jobtitle.csv')
        expected_count = 'test,2'
        assert key.get_contents_as_string().decode('utf-8').rstrip() == expected_count


def test_field_value_counter_lambda():
    def extractSalaryRange(document):
        return '-'.join([
            str(document.get('baseSalary', {}).get('minValue', '?')),
            str(document.get('baseSalary', {}).get('maxValue', '?'))
        ])
    counter = FieldValueCounter(
        quarter='2014Q1',
        field_values=[
            ('stringSalaryRange', extractSalaryRange)
        ]
    )
    counter.track(
        input_document={
            'baseSalary': {
                'maxValue': 0.0,
                '@type': 'MonetaryAmount',
                'minValue': 0.0
            }
        }
    )
    counter.track(
        input_document={
            'baseSalary': {
                'maxValue': '$5',
                '@type': 'MonetaryAmount',
                'minValue': '$5'
            }
        }
    )
    counter.track(
        input_document={
            'baseSalary': {
                '@type': 'MonetaryAmount',
                'minValue': '$5'
            }
        }
    )
    assert counter.accumulator['stringSalaryRange']['0.0-0.0'] == 1
    assert counter.accumulator['stringSalaryRange']['$5-$5'] == 1
    assert counter.accumulator['stringSalaryRange']['$5-?'] == 1


def test_field_value_counter_listresult():
    def extractSkills(document):
        baseSkills = document.get('skills', '')
        if isinstance(baseSkills, list):
            return baseSkills
        else:
            return [skill.strip() for skill in baseSkills.split(',')]
    counter = FieldValueCounter(
        quarter='2014Q1',
        field_values=[
            ('rawSkills', extractSkills)
        ]
    )
    counter.track(
        input_document={
            'skills': 'Customer Service, Consultant, Entry Level'
        }
    )
    counter.track(
        input_document={
            'skills': ['Slicing', 'Dicing', 'Entry Level']
        }
    )
    assert counter.accumulator['rawSkills']['Entry Level'] == 2
    assert counter.accumulator['rawSkills']['Customer Service'] == 1
    assert counter.accumulator['rawSkills']['Consultant'] == 1
    assert counter.accumulator['rawSkills']['Slicing'] == 1
    assert counter.accumulator['rawSkills']['Dicing'] == 1
