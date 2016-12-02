"""
Utilities used by airflow operators and DAGs
"""
import math


def datetime_to_quarter(dt):
    """
    Args:
        dt: a datetime
    Returns:
        the datetime's quarter in string format (2015Q1)
    """
    year = dt.year
    quarter = int(math.ceil(float(dt.month)/3))
    return '{}Q{}'.format(year, quarter)
