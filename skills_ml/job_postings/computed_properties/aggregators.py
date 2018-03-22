"""Aggregate job posting computed properties into tabular datasets"""
import s3fs

import logging
import pandas as pd
import textwrap
from skills_utils.time import quarter_to_daterange, dates_in_range


def df_for_properties_and_dates(computed_properties, dates):
    """Assemble a dataframe with the raw data from many computed properties and dates

    Args:
        computed_properties (list of JobPostingComputedProperty)
        dates (list of datetime.date objects)

    Returns: pandas.DataFrame
    """
    datestrings = [date.strftime("%Y-%m-%d") for date in dates]
    dataframes = [
        computed_property.df_for_dates(datestrings)
        for computed_property in computed_properties
    ]
    return dataframes[0].join(dataframes[1:])


def expand_array_col_to_many_cols(base_col, func, aggregation):
    """Expand an array column created as the result of an .aggregate call into many columns

    Args:
        base_col (string) The name of the base column (before .aggregate)
        func (function) The base function that was aggregated on
        aggregation (pandas.DataFrame) The post-aggregation dataframe

    Returns: pandas.DataFrame, minus the array column and plus columns for each array value
    """
    logging.info('Expanding array column %s to multiple columns', base_col)
    # base_col + function name is how pandas names columns
    # that are the result of aggregate functions
    full_col_name = '_'.join([base_col, func.__name__])

    # procedure for expansion:
    # 1. make each row a pandas Series, which will be the right shape but with in names
    # 2. Prepend the base column name to the name of each
    # 3. Turn the NaNs for shorter lists into empty strings
    # 4. Drop the old column from the aggregation and join with the new df
    new_cols = aggregation[full_col_name].apply(pd.Series)
    new_cols = new_cols.add_prefix(base_col + '_').fillna('')
    return aggregation.drop(full_col_name, axis=1).join(new_cols)


def base_func(aggregate_function):
    """Deals with the possibility of functools.partial being applied to a given
    function. Allows access to the decorated 'return' attribute whether or not
    it is also a partial function

    Args:
        aggregate_function (callable) Either a raw function or a functools.partial object

    Returns: callable
    """
    logging.info('Checking base func for %s', aggregate_function)
    if hasattr(aggregate_function, 'func'):
        return aggregate_function.func
    else:
        return aggregate_function


def validate_aggregate_functions(aggregate_properties, aggregate_functions):
    for key, value in aggregate_functions.items():
        logging.info('Validating aggregate functions for column %s', key)
        if not isinstance(value, list):
            raise ValueError(
                'aggregate_functions values must all be lists. Found non-list at {}'.format(key)
            )
        for func in value:
            logging.info('Validating aggregate function %s', func)
            found = False
            logging.info('Finding aggregate property that produces %s ', key)
            for aggregate_property in aggregate_properties:
                for col in aggregate_property.property_columns:
                    logging.info('Checking column name %s', col.name)
                    if key == col.name:
                        found = True
                        logging.info('Found home for %s. But is it listed as compatible?', key)
                        if all(base_func(func).__qualname__ not in path for path in col.compatible_aggregate_function_paths):
                            raise ValueError(textwrap.dedent('''
                                Aggregate function {} appears to be paired with computed property {}
                                but is not on list of compatible aggregate functions for column: {}
                            '''.format(
                                base_func(func).__qualname__,
                                aggregate_property.__class__.__name__,
                                col.compatible_aggregate_function_paths
                            )))
            if not found:
                raise ValueError(textwrap.dedent('''
                    Aggregate function {} does not appear to be paired with any
                    present computed property'''.format(
                        base_func(func).__qualname__
                    )
                ))
    logging.info('Validation complete')


def aggregation_for_properties_and_dates(
    grouping_properties,
    aggregate_properties,
    aggregate_functions,
    dates
):
    """Assemble an aggregation dataframe for given dates

    Args:
        grouping_properties (list of JobPostingComputedProperty)
            Properties to form the primary key of the aggregation
        aggregate_properties (list of JobPostingComputedProperty)
            Properties to be aggregated over the primary key
        aggregate_functions (dict) A lookup of aggregate functions
            to be applied for each aggregate column
        dates (list of datetime.date) The desired dates for the aggregation to cover

    Returns: pandas.DataFrame indexed on the grouping properties,
        covering all data from the given dates
    """
    validate_aggregate_functions(aggregate_properties, aggregate_functions)
    big_df = df_for_properties_and_dates(grouping_properties + aggregate_properties, dates)
    grouping_column_lists = [p.property_columns for p in grouping_properties]
    grouping_column_names = [col.name for collist in grouping_column_lists for col in collist]
    aggregation = big_df.groupby(grouping_column_names).agg(aggregate_functions)
    # Since the aggregate_functions values is a list
    # the column names will be lists as well
    # one part being the base column and the other being the aggregate function
    aggregation.columns = ['_'.join(t) for t in aggregation.columns]

    # Some functions (like skill extractors) will return lists of items,
    # often representing the top n items. We want to expand these array columns
    # into multiple columns
    for key, funclist in aggregate_functions.items():
        for aggregate_function in funclist:
            func = base_func(aggregate_function)
            if hasattr(func, 'returns') and func.returns == 'list':
                aggregation = expand_array_col_to_many_cols(key, func, aggregation)
    return aggregation


def aggregate_properties_for_quarter(
    quarter,
    grouping_properties,
    aggregate_properties,
    aggregate_functions,
    aggregations_path,
    aggregation_name
):
    """Aggregate computed properties for a quarter and writes the resulting CSV to S3

    Args:
        quarter (string)
        grouping_properties (list of JobPostingComputedProperty)
            Properties to form the primary key of the aggregation
        aggregate_properties (list of JobPostingComputedProperty)
            Properties to be aggregated over the primary key
        aggregate_functions (dict) A lookup of aggregate functions
            to be applied for each aggregate column
        aggregations_path (string) The base s3 path to store aggregations
        aggregation_name (string) The name of this particular aggregation

    Returns: nothing
    """
    start_date, end_date = quarter_to_daterange(quarter)
    included_dates = [date for date in dates_in_range(start_date, end_date)]
    aggregation_df = aggregation_for_properties_and_dates(
        grouping_properties,
        aggregate_properties,
        aggregate_functions,
        included_dates
    )

    out_path = '/'.join([
        aggregations_path,
        aggregation_name,
        quarter + '.csv'
    ])
    fs = s3fs.S3FileSystem()
    with fs.open(out_path, 'wb') as f:
        f.write(aggregation_df.to_csv(None).encode())
    return out_path
