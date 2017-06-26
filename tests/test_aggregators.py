from skills_ml.algorithms.aggregators import CountAggregator


def test_addition():
    first_aggregator = CountAggregator()
    second_aggregator = CountAggregator()

    first_aggregator.accumulate(job_posting='1', job_key=('1',), groups=['1'])
    second_aggregator.accumulate(job_posting='2', job_key=('2',), groups=['2'])

    first_aggregator += second_aggregator

    assert first_aggregator.group_values == {
        ('1', ('1',)): {'total': 1},
        ('2', ('2',)): {'total': 1},
    }
