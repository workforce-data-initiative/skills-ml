"""Filtering streamed job postings"""

from .common_schema import JobPostingType, JobPostingGeneratorType, MetadataType

from typing import Callable, List


def soc_major_group_filter(major_groups: List) -> Callable:
    """Return a function that checks the ONET Soc Code of a job posting (if it is present) against the configured major groups.
    """
    def job_posting_is_in_major_group(document: JobPostingType) -> bool:
        key = 'onet_soc_code'
        if not document[key]:
            return False
        if document[key][:2] not in major_groups:
            return False
        return True
    return job_posting_is_in_major_group


class JobPostingFilterer(object):
    """Filter common schema job postings through a number of filtering functions

    Args:
        job_posting_generator: An iterable of job postings (each in dict form)
        filter_funcs: A list of filtering functions, each taking in a job posting document (as dict) and returning a boolean instructing whether or not the posting passes the filter
    """
    def __init__(
        self,
        job_posting_generator: JobPostingGeneratorType,
        filter_funcs: List[Callable]
    ):
        self.job_posting_generator = job_posting_generator
        self.filter_funcs = filter_funcs

    def __iter__(self) -> JobPostingGeneratorType:
        """Yield all job postings in the configured generator that pass all configured filtering functions"""
        for job_posting in self.job_posting_generator:
            can_yield = all(filter_func(job_posting) for filter_func in self.filter_funcs)
            if can_yield:
                yield job_posting

    @property
    def metadata(self) -> MetadataType:
        return {'job_posting_filters': ':'.join(str(func) for func in self.filter_funcs)}
