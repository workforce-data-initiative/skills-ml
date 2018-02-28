"""Test job normalizers

Requires 'interesting_job_titles.csv' to be populated, of format:
input job title\tdescription of job\tONET code

Each task will output two CSV files, one with the normalizer's ranks
and one without ranks. The latter is for sending to people to fill out
and the former is for testing those results against the normalizer's

Originally written by Kwame Porter Robinson
"""

from abc import ABCMeta, abstractmethod
import csv
import pandas as pd
import json
import random
import requests

from skills_ml.algorithms.job_normalizers import esa_jobtitle_normalizer

from enum import IntEnum


class InputSchema(IntEnum):
    """An enumeration listing the data elements and indices taken from source data"""
    job_title = 0
    description = 1
    soc_code = 2


class InterimSchema(IntEnum):
    """An enumeration listing the data elements and indices after normalization"""
    job_title = 0
    description = 1
    soc_code = 2
    normalizer_response = 3


class NormalizerResponse(metaclass=ABCMeta):
    """
    Abstract interface for enforcing common iteration, access patterns
    to a variety of possible normalizers.

    Args:
        name (string): A name for the normalizer
        access (filename or file object): A tab-delimited CSV with column order {job_title, description, soc_code}
        num_examples (int, optional): Number of top responses to include

    Normalizers should return a list of results, ordered by relevance,
    with 'title' and optional 'relevance_score' keys
    """

    def __init__(self, name=None, access=None, num_examples=3):
        self.name = name
        self.access = access
        self.num_examples = num_examples

    def __iter__(self):
        """
        Iterate through the input file and yield the original
        inputs along with the normalizer response
        """
        iter_obj = self._access()
        for key, item in iter_obj:
            row = [None] * len(InterimSchema)
            row[InterimSchema.job_title] = item[InputSchema.job_title]
            row[InterimSchema.description] = item[InputSchema.description]
            row[InterimSchema.soc_code] = item[InputSchema.soc_code]
            row[InterimSchema.normalizer_response] = \
                self.normalize(item[InputSchema.job_title])
            yield row

    def _access(self):
        """
        Opens up an iterator over the *data stream* to normalize
        Uses self.access to initalize/locate stream
        """
        return pd.read_csv(self.access,
                           sep='\t',
                           header=None).iterrows()

    @abstractmethod
    def normalize(self, job_title):
        """
        Gets response from normalizer
        """
        pass

    @abstractmethod
    def _good_response(self, response):
        """
        Returns a boolean describing whether or not the normalizer
        response is usable
        """
        pass

    def ranked_rows(self, response):
        """
            Parses a normalizer response for one job title,
            reshuffles the top responses (defined by self.num_examples),
            and yields a flat representation for each
        """
        if self._good_response(response):
            desc = response[InterimSchema.description]
            normalizer_results = response[InterimSchema.normalizer_response]
            jobtitle = response[InterimSchema.job_title]
            normalized_responses = [
                (jobtitle, desc, norm_response['title'], i)
                for i, norm_response
                in enumerate(normalizer_results[0:self.num_examples])
            ]
            random.shuffle(normalized_responses)
            for row in normalized_responses:
                yield row


class MiniNormalizer(NormalizerResponse):
    """
    Access normalizer classes which can be instantiated and
    implement 'normalize_job_title(job_title)'
    """
    def __init__(self, name, access, normalize_class):
        super().__init__(name, access)
        self.normalizer = normalize_class()

    def normalize(self, job_title):
        return self.normalizer.normalize_job_title(job_title)

    def _good_response(self, response):
        return len(response) > 0 and \
            len(response[InterimSchema.normalizer_response]) > 0


class DataAtWorkNormalizer(NormalizerResponse):
    endpoint_url = r"http://api.dataatwork.org/v1/jobs/normalize"

    def normalize(self, job_title):
        response = requests.get(
            self.endpoint_url,
            params={'job_title': job_title, 'limit': self.num_examples}
        )
        return response.json()

    def _good_response(self, response):
        return 'error' not in response[InterimSchema.normalizer_response]


def generate_evaluators(evaluation_filename):
    return [
        (MiniNormalizer, {
            'name': 'Explicit_Semantic_Analysis_Normalizer',
            'access': evaluation_filename,
            'normalize_class': esa_jobtitle_normalizer.ESANormalizer
        }),
        (DataAtWorkNormalizer, {
            'name': 'Elasticsearch_API_Normalizer',
            'access': evaluation_filename
        })
    ]


def run_evaluator(evaluator_class, **kwargs):
    evaluator = evaluator_class(**kwargs)
    filename = '{}_output.csv'.format(evaluator.name)
    unranked_filename = '{}_unranked_output.csv'.format(evaluator.name)
    with open(filename, 'w') as f:
        with open(unranked_filename, 'w') as uf:
            writer = csv.writer(f)
            unranked_writer = csv.writer(uf)
            unranked_writer.writerow([
                'interesting job title',
                'short job desc',
                'normalized job title',
                'rank relevance of normalized job title (-1 for irrelevant)'
            ])
            for response in evaluator:
                for ranked_row in evaluator.ranked_rows(response):
                    writer.writerow(ranked_row)
                    unranked_writer.writerow(ranked_row[:-1])
