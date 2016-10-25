from abc import ABCMeta, abstractmethod
import csv
import pandas as pd
import json
import random
import requests

from algorithms.job_normalizers import esa_jobtitle_normalizer
from airflow import DAG
from airflow.operators import DummyOperator, PythonOperator
from datetime import datetime, timedelta

"""Test job normalizers

Requires 'interesting_job_titles.csv' to be populated, of format:
input job title\tdescription of job\tONET code

Each task will output two CSV files, one with the normalizer's ranks
and one without ranks. The latter is for sending to people to fill out
and the former is for testing those results against the normalizer's
"""


class NormalizerResponse(metaclass=ABCMeta):
    """
    Abstract interface for enforcing common iteration, access patterns
    to a variety of possible normalizers.
    """
    def __init__(self, name=None, access=None, num_examples=3):
        self.name = name
        self.access = access
        self.num_examples = num_examples

    def __iter__(self):
        iter_obj = self._access()
        for key, item in iter_obj:
            yield self._get_response((item[1], item[2]),
                                     item[0])

    def _access(self):
        """
        Opens up an iterator over the *data stream* to normalize
        Uses self.access to initalize/locate stream
        """
        return pd.read_csv(self.access,
                           sep='\t',
                           header=None).iterrows()

    def _get_response(self, answer, job_title):
        """
        Gets response from normalizer when provided item(s)
        from the data stream (job titles)
        """
        return (answer, job_title, self.normalize(job_title))

    @abstractmethod
    def _good_response(response):
        pass

    def ranked_rows(self, response):
        if self._good_response(response):
            normalized_responses = [
                (response[1], response[0][0], norm_response['title'], i)
                for i, norm_response
                in enumerate(response[2][0:self.num_examples])
            ]
            random.shuffle(normalized_responses)
            for row in normalized_responses:
                yield row


class MiniNormalizer(NormalizerResponse):
    def __init__(self, name, access, normalize_class):
        super().__init__(name, access)
        self.normalizer = normalize_class()

    def normalize(self, job_title):
        return self.normalizer.normalize_job_title(job_title)

    def _good_response(self, response):
        return len(response) > 0 and len(response[2]) > 0


class DataAtWorkNormalizer(NormalizerResponse):
    endpoint_url = r"http://api.dataatwork.org/v1/jobs/normalize"

    def normalize(self, job_title):
        response = requests.get(
            self.endpoint_url,
            params={'job_title': job_title, 'limit': self.num_examples}
        )
        return response.json()

    def _good_response(self, response):
        return 'error' not in response[2]


def instantiate_evaluators(evaluation_filename):
    return [
        MiniNormalizer(
            name='Explicit_Semantic_Analysis_Normalizer',
            access=evaluation_filename,
            normalize_class=esa_jobtitle_normalizer.ESANormalizer
        ),
        DataAtWorkNormalizer(
            name='Elasticsearch_API_Normalizer',
            access=evaluation_filename
        )
    ]


def run_evaluator(evaluator=None):
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

# some DAG args, please tweak for sanity
default_args = {
    'evaluation_file': 'interesting_job_titles.csv',
    'owner': 'job_title_normalizer',
    'depends_on_past': True,
    'start_date': datetime.today()
}

dag = DAG('job_title_normalizer_evaluation',
          schedule_interval=None,
          default_args=default_args)

run_this = DummyOperator(
    task_id='Root',
    dag=dag)

for evaluator in instantiate_evaluators(default_args['evaluation_file']):
    task = PythonOperator(
        task_id=evaluator.name,
        python_callable=run_evaluator,
        op_kwargs={'evaluator': evaluator},
        dag=dag)
    task.set_upstream(run_this)
