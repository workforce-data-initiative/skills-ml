"""Encapsulates the computation of some piece of data for job postings, to make aggregation
and tabular datasets easy to produce"""
from abc import ABCMeta, abstractmethod
import logging

import pandas

from skills_ml.storage  import PersistedJSONDict

class JobPostingComputedProperty(metaclass=ABCMeta):
    """Base class for computers of job posting properties.

    Using this class, expensive computations can be performed once, stored on S3 per job posting
    in daily partitions, and reused in different aggregations.

    The base class takes care of all of the serialization and partitioning,
    leaving subclasses to implement a function for computing the property of a single posting
    and metadata describing the output of this function.

    Subclasses must implement:
        - _compute_func_on_one to produce a callable that takes in a single
            job posting and returns JSON-serializable output representing the computation target.
            This function can produce objects that are kept in scope and reused,
            so properties that require a large object (e.g. a trained classifier) to do their
            computation work can be downloaded from S3 here without requiring the I/O work
            to be done over and over. (See .computers.SOCClassifyProperty for illustration)
        - property_name attribute (string) that is used when saving the computed properties
        - property_columns attribute (list) of ComputedPropertyColumns that
            map to the column names output by `_compute_func_on_one`

    Args:
        path (string) An s3 base path to store the properties.
            The caches will be namespaced by the property name and date
    """
    def __init__(self, storage):
        self.storage = storage

    def compute_on_collection(self, job_postings_generator):
        """Compute and save to s3 a property for every job posting in a collection.

        Will save data keyed on the job posting ID in JSON format to S3,
        so the output of the property must be JSON-serializable.

        Partitions data daily, according to the datePosted of the job posting.

        Args:
            job_postings_generator (iterable of dicts) Any number of job postings,
                each in dict format
        """
        caches = {}
        misses = 0
        hits = 0
        property_computer = self._compute_func_on_one()
        for number, job_posting in enumerate(job_postings_generator):
            if number % 1000 == 0:
                logging.info(
                    'Computation of job posting properties for %s on posting %s',
                    self.property_name,
                    number
                )
            date_posted = job_posting.get('datePosted', 'unknown')
            if date_posted not in caches:
                caches[date_posted] = self.cache_for_date(date_posted)
            if job_posting['id'] not in caches[date_posted]:
                caches[date_posted][job_posting['id']] = property_computer(job_posting)
                misses += 1
            else:
                hits += 1
        for cache in caches.values():
            cache.save()
        logging.info(
            'Computation of job posting properties for %s complete. %s hits, %s misses',
            self.property_name,
            hits,
            misses
        )

    def cache_for_date(self, datestring):
        """Return the property cache for a given date

        Args:
            datestring (string): A string representing the date in the S3 path
        Returns: (skills_utils.s3.S3BackedJsonDict)
        """
        fname = '/'.join([self.property_name, datestring]) + '.json'
        return PersistedJSONDict(self.storage, fname)

    def df_for_date(self, datestring):
        """Return a dataframe of the cached data for a given date

        Args:
            datestring (string): A string representing the date in the S3 path
        Returns: (pandas.DataFrame)
        """
        cache = self.cache_for_date(datestring)
        if len(cache) == 0:
            return None
        df = pandas.DataFrame.from_dict(cache, orient='index')
        if len(df.columns) != len(self.property_columns):
            raise ValueError('''Length of self.property_columns for this class ({})
            does not equal the length of columns in the produced dataframe ({})'''.format(
                len(self.property_columns),
                len(df.columns)
            ))
        df.columns = [col.name for col in self.property_columns]
        logging.info('Dataframe for %s = %s total rows', datestring, len(df))
        return df

    def df_for_dates(self, datestrings):
        """Return a dataframe representing all values from the given dates

        Args:
            datestrings (list): A list of date strings

        Returns: (pandas.DataFrame)
        """
        logging.info('Computing dataframe for %s dates', len(datestrings))
        df = pandas.concat(self.df_for_date(datestring) for datestring in datestrings)
        logging.info('Final dataframe = %s total rows', len(df))
        return df

    @abstractmethod
    def _compute_func_on_one(self):
        """Return a function that takes in a job posting and compute the property value

        Returns: callable
        """
        pass


class ComputedPropertyColumn(object):
    """Metadata about a specific output column of a computed property

    Args:
        name (string) The name of the column
        description (string) A description of the column and how it was populated.
        compatible_aggregate_function_paths (dict, optional): If this property is meant to be
            used in aggregations, map string function paths to descriptions of what the
            function is computing for this column.
            All function paths should be compatible with pandas.agg (one argument, an iterable),
            though multi-argument functions can be used in conjunction with functools.partial
    """
    def __init__(self, name, description, compatible_aggregate_function_paths=None):
        self.name = name
        self.description = description
        self.compatible_aggregate_function_paths = compatible_aggregate_function_paths
