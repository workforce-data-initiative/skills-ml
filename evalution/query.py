from abc import ABCMeta, abstractmethod
import pandas as pd
import esa_jobtitle_normalizer
import json
import requests

from airflow import DAG
from airflow.operators import PythonOperator
from datetime import datetime, timedelta

class NormalizerResponse(metaclass=ABCMeta):
    """
    Abstract interface for enforcing common iteration, access patterns
    to a variety of possible normalizers.
    """
    def __init__(self, name=None, access=None, logger=None):
        self.name = name
        self.access = access # initalizing object for access test source data to push to normalizer
        self.logger = logger # object for pushing normalizer response to some store (ES, local file,etc)
    
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def _access(self):
        """
        Opens up an iterator over the *data stream* to normalize
        Uses self.access to initalize/locate stream
        """
        pass

    @abstractmethod
    def _get_response(self):
        """
        Gets response from normalizer when provided item(s) from the data stream (job titles)
        """
        pass

    @abstractmethod
    def log_response(self):
        pass

class Mini_Normalizer(NormalizerResponse):
    def __init__(self, name, access, normalize):
        super().__init__(name, access)
        self.normalize = normalize

    def __iter__(self):
        iter_obj = self._access()
        for key, item in iter_obj:
            yield self._get_response((item[1], item[2]),
                                     item[0])

    def _access(self):
        return pd.read_csv(self.access,
                           sep='\t',
                           header=None).iterrows()

    def _get_response(self, answer, job_title):
        return (answer, job_title, self.normalize(job_title))

    def log_response(self, response):
        print(json.dumps({'response': response[2],
                          'job_title': response[1],
                          'description': response[0][0],
                          'SOC*Code': response[0][1]}))

class API_Normalizer(NormalizerResponse):
    def __init__(self, name, access, endpoint_url, normalize=None):
        super().__init__(name, access)
        self.endpoint_url = endpoint_url
        self.normalize = self._get_job_normalize

    def _get_job_normalize(self, job_title):
        r = requests.get(self.endpoint_url, params={'job_title': job_title})
        return r.json()

    def __iter__(self):
        iter_obj = self._access()
        for key, item in iter_obj:
            yield self._get_response((item[1], item[2]),
                                     item[0])

    def _access(self):
        return pd.read_csv(self.access,
                           sep='\t',
                           header=None).iterrows()

    def _get_response(self, answer, job_title):
        return (answer, job_title, self.normalize(job_title))

    def log_response(self, response):
        print(json.dumps({'response': response[2],
                          'job_title': response[1],
                          'description': response[0][0],
                          'SOC*Code': response[0][1]}))

def instantiate_evaluators(access=access):# should probably borrow more from default args?
    normalizers_to_evaluate = [Mini_Normalizer(name='Explicit Semantic Analysis Normalizer',
                                access=access,
                                normalize = esa_jobtitle_normalizer.normalize_job_title),
                               API_Normalizer(name='Elasticsearch/API Normalizer',
                                access=access,
                                endpoint_url = r"http://api.dataatwork.org/v1/jobs/normalize")]
    return normalizers_to_evaluate

def run_evaluator(evaluator=None):
    for response in evaluator:
        evaluator.log_response(response)

# some DAG args, please tweak for sanity
default_args = {
    'evaluation_file': 'small_job_titles.csv',
    'owner':'job_title_normalizer',
    'depends_on_past':True,
    'start_date': datetime.today()
}

dag = DAG('job_title_normalizer_evaluation',
          schedule_interval='@once',
          default_args=default_args)

run_this = DummyOperator(
        task_id='Root ...',
        dag=dag)

for evaluator in instantiate_evaluators(default_args['evaluation_file']):
    task = PythonOperator(
            task_id= evaluator.name,
            python_callable=run_evaluator,
            op_kwards={'evaluator':evaluator},
            dag=dag)
    task.set_upstream(run_this)

##print(k.normalize( job_title) )
##print(h.normalize( job_title) )
#for e in h:
#    h.log_response(e)
#print('\n\n')
#for e in k:
#    k.log_response(e)
