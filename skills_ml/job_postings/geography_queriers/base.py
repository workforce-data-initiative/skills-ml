from abc import ABCMeta, abstractmethod


class JobGeographyQuerier(metaclass=ABCMeta):
    """Base class for retrievers/computers of geography data from job postings

    The main interface is query(job_posting), which returns
    a tuple the same length as self.output_columns.

    Subclasses must implement:
        output_columns (property/attribute): a collection of two-tuples with a name and description
            for each column output by the querier
        name (property/attribute) a name of the querier
        _query(job_posting) to take a job posting object and return
            a tuple of the same length as self.output_columns.
    """
    @property
    @abstractmethod
    def output_columns(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def _query(self, job_posting):
        pass

    def query(self, job_posting):
        result = self._query(job_posting)
        assert len(result) == len(self.output_columns)
        return result
