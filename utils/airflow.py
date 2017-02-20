"""
Utilities used by airflow operators and DAGs
"""
import math


def datetime_to_year_quarter(dt):
    """
    Args:
        dt: a datetime
    Returns:
        tuple of the datetime's year and quarter
    """
    year = dt.year
    quarter = int(math.ceil(float(dt.month)/3))
    return (year, quarter)


def datetime_to_quarter(dt):
    """
    Args:
        dt: a datetime
    Returns:
        the datetime's quarter in string format (2015Q1)
    """
    year, quarter = datetime_to_year_quarter(dt)
    return '{}Q{}'.format(year, quarter)
