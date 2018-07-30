"""Encapsulates the computation of some piece of data for job postings, to make aggregation
and tabular datasets easy to produce"""
from abc import ABCMeta, abstractmethod
import logging

import pandas

from skills_ml.storage import PersistedJSONDict


class JobPostingComputedProperty(metaclass=ABCMeta):
    """Base class for computers of job posting properties.

    Using this class, expensive computations can be performed once, stored on S3 per job posting
    in partitions, and reused in different aggregations.

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
        storage (skills_ml.storage.Store) A storage object in which to store the cached properties.
        partition_func (callable, optional) A function that takes a job posting and
            outputs a string that should be used as a partition key. Must be deterministic.
            Defaults to the 'datePosted' value

            The caches will be namespaced by the property name and partition function
    """
    def __init__(self, storage, partition_func=None):
        self.storage = storage
        if not partition_func:
            self.partition_func = lambda posting: posting['datePosted']
        else:
            self.partition_func = partition_func

    def compute_on_collection(self, job_postings_generator):
        """Compute and save to s3 a property for every job posting in a collection.

        Will save data keyed on the job posting ID in JSON format to S3,
        so the output of the property must be JSON-serializable.

        Partitions data according to the output of the property's partition_func on the posting

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
            partition_key = self.partition_func(job_posting)
            if partition_key not in caches:
                caches[partition_key] = self.cache_for_key(partition_key)
            if job_posting['id'] not in caches[partition_key]:
                caches[partition_key][job_posting['id']] = property_computer(job_posting)
                misses += 1
                if misses > 0 and misses % 100000 == 0:
                    logging.info('Proactively saving caches at cache miss %s (job posting %s)', misses, number)
                    for cache in caches.values():
                        cache.save()
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

    def cache_for_key(self, key):
        """Return the property cache for a given key

        Args:
            key (string): The partition key
        Returns: (skills_ml.storage.PersistedJSONDict)
        """
        fname = '/'.join([self.property_name, key]) + '.json'
        return PersistedJSONDict(self.storage, fname)

    def cache_keys(self):
        """A list of cache keys used by this property.

        Computed by querying the storage object for all existing caches

        Returns: (list of strings) all known partition keys, each of which is
            recognized by the various *_for_key methods
        """
        matching_files = self.storage.list(self.property_name)
        return [f.replace('.json', '') for f in matching_files if f.endswith('.json')]

    def df_for_key(self, key):
        """Return a dataframe of the cached data for a given key

        Args:
            key (string): The partition key
        Returns: (pandas.DataFrame)
        """
        cache = self.cache_for_key(key)
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
        logging.info('Dataframe for %s = %s total rows', key, len(df))
        return df

    def df_for_keys(self, keys):
        """Return a dataframe representing all values from the given dates

        Args:
            datestrings (list): A list of date strings

        Returns: (pandas.DataFrame)
        """
        logging.info('Computing dataframe for %s dates', len(keys))
        df = pandas.concat(self.df_for_key(key) for key in keys)
        logging.info('Final dataframe = %s total rows', len(df))
        return df

    @abstractmethod
    def _compute_func_on_one(self):
        """Return a function that takes in a job posting and compute the property value

        Returns: callable
        """
        pass

    @property
    @abstractmethod
    def property_columns(self):
        pass

    @property
    @abstractmethod
    def property_name(self):
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
