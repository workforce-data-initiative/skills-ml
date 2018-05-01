from skills_ml.job_postings.computed_properties import JobPostingComputedProperty, ComputedPropertyColumn
from skills_ml.job_postings.computed_properties.aggregators import aggregation_for_properties_and_dates, aggregate_properties_for_quarter
from skills_ml.job_postings.aggregate.pandas import listy_n_most_common
from skills_ml.storage import S3Store
import datetime
from functools import partial
import numpy
from moto import mock_s3
import boto3
import s3fs
import unicodecsv as csv


class FakeComputedProperty(JobPostingComputedProperty):
    def __init__(self):
        pass

    def _compute_func_on_one(self):
        raise ValueError('Should not be calling compute on this class')


class FakeGroupingPropertyOne(FakeComputedProperty):
    property_name = 'grouping_property_one'
    property_columns = [ComputedPropertyColumn(name='grouping_property_one', description='')]

    def cache_for_date(self, datestring):
        if datestring == '2015-04-01':
            return {'1': 'value1', '2': 'value2'}
        elif datestring == '2015-04-02':
            return {'3': 'value3', '4': 'value4'}
        else:
            return {}


class FakeGroupingPropertyTwo(FakeComputedProperty):
    property_name = 'grouping_property_two'
    property_columns = [ComputedPropertyColumn(name='grouping_property_two', description='')]

    def cache_for_date(self, datestring):
        if datestring == '2015-04-01':
            return {'1': 'value5', '2': 'value6'}
        elif datestring == '2015-04-02':
            return {'3': 'value5', '4': 'value6'}
        else:
            return {}


class FakeAggregationPropertyOne(FakeComputedProperty):
    property_name = 'aggregation_property_one'
    property_columns = [
        ComputedPropertyColumn(
            name='aggregation_property_one',
            description='',
            compatible_aggregate_function_paths={
                'skills_ml.algorithms.aggregators.pandas.listy_n_most_common': ''}
        )
    ]

    def cache_for_date(self, datestring):
        if datestring == '2015-04-01':
            return {'1': [['1', '1', '3']], '2': [['1', '3', '3']]}
        elif datestring == '2015-04-02':
            return {'3': [['4', '4', '4']], '4': [['4', '4', '6']]}
        else:
            return {}


class FakeAggregationPropertyTwo(FakeComputedProperty):
    property_name = 'aggregation_property_two'
    property_columns = [
        ComputedPropertyColumn(
            name='aggregation_property_two',
            description='',
            compatible_aggregate_function_paths={'numpy.sum': ''}
        )
    ]

    def cache_for_date(self, datestring):
        if datestring == '2015-04-01':
            return {'1': 1, '2': 1}
        elif datestring == '2015-04-02':
            return {'3': 1, '4': 1}
        else:
            return {}


def test_aggregation_for_properties_and_dates():
    aggregation = aggregation_for_properties_and_dates(
        grouping_properties=[FakeGroupingPropertyOne(), FakeGroupingPropertyTwo()],
        aggregate_properties=[FakeAggregationPropertyOne(), FakeAggregationPropertyTwo()],
        aggregate_functions={'aggregation_property_two': [numpy.sum], 'aggregation_property_one': [partial(listy_n_most_common, 2)]},
        dates=[datetime.date(2015, 4, 1), datetime.date(2015, 4, 2)]
    )
    assert len(aggregation) == 4
    assert sorted(aggregation.columns) == [
        'aggregation_property_one_0',
        'aggregation_property_one_1',
        'aggregation_property_two_sum'
    ]
    assert sorted(aggregation.index.names) == ['grouping_property_one', 'grouping_property_two']
    assert aggregation.reset_index().to_dict('records') == [
        {
            'grouping_property_one': 'value1',
            'grouping_property_two': 'value5',
            'aggregation_property_one_0': '1',
            'aggregation_property_one_1': '3',
            'aggregation_property_two_sum': 1,
        },
        {
            'grouping_property_one': 'value2',
            'grouping_property_two': 'value6',
            'aggregation_property_one_0': '3',
            'aggregation_property_one_1': '1',
            'aggregation_property_two_sum': 1,
        },
        {
            'grouping_property_one': 'value3',
            'grouping_property_two': 'value5',
            'aggregation_property_one_0': '4',
            'aggregation_property_one_1': '',
            'aggregation_property_two_sum': 1,
        },
        {
            'grouping_property_one': 'value4',
            'grouping_property_two': 'value6',
            'aggregation_property_one_0': '4',
            'aggregation_property_one_1': '6',
            'aggregation_property_two_sum': 1,
        },
    ]

@mock_s3
def test_aggregate_properties_in_quarter():
    client = boto3.resource('s3')
    client.create_bucket(Bucket='test-bucket')
    s3_storage = S3Store('s3://test-bucket/aggregations')
    aggregate_properties_for_quarter(
        '2015Q2',
        grouping_properties=[FakeGroupingPropertyOne(), FakeGroupingPropertyTwo()],
        aggregate_properties=[FakeAggregationPropertyOne(), FakeAggregationPropertyTwo()],
        aggregate_functions={'aggregation_property_two': [numpy.sum], 'aggregation_property_one': [partial(listy_n_most_common, 2)]},
        storage=s3_storage,
        aggregation_name='fake_agg'
    )
    s3 = s3fs.S3FileSystem()
    with s3.open('s3://test-bucket/aggregations/fake_agg/2015Q2.csv', 'rb') as f:
        reader = csv.reader(f)
        num_rows = len([row for row in f])
        assert num_rows == 5
