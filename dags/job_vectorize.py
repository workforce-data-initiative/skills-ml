import csv
import logging
import os
from datetime import datetime
from calendar import monthrange
import json
import pandas as pd

from airflow import DAG
from airflow.hooks import S3Hook
from airflow.operators import BaseOperator

from utils.airflow import datetime_to_quarter
from utils.s3 import upload
from utils.hash import md5
from config import config
from datasets import job_postings

from algorithms.corpus_creators.basic import GensimCorpusCreator
from algorithms.job_vectorizers.doc2vec_vectorizer import Doc2Vectorizer

# some DAG args, please tweak for sanity
default_args = {
    'depends_on_past': False,
    'start_date': datetime(2011, 4, 1),
}

dag = DAG(
    'job_vectorizer',
    schedule_interval=None,
    default_args=default_args
)

from datetime import date
from calendar import monthrange

def quarter_boundaries(quarter):
    year, quarter = quarter.split('Q')
    year = int(year)
    quarter = int(quarter)
    first_month_of_quarter = 3 * quarter - 2
    last_month_of_quarter = 3 * quarter
    first_day = date(year, first_month_of_quarter, 1)
    last_day = date(year, last_month_of_quarter, monthrange(year, last_month_of_quarter)[1])
    return first_day, last_day


def metta_config(quarter, num_dimensions):
    first_day, last_day = quarter_boundaries(quarter)
    return {
        'start_time': first_day,
        'end_time': last_day,
        'prediction_window': 3, # ???
        'label_name': 'onet_soc_code',
        'label_type': 'categorical',
        'matrix_id': 'job_postings_{}'.format(quarter),
        'feature_names': ['doc2vec_{}'.format(i) for i in range(num_dimensions)],
    }


class JobVectorizeOperator(BaseOperator):
    def execute(self, context):
        s3_conn = S3Hook().get_conn()
        quarter = datetime_to_quarter(context['execution_date'])
        job_vector_filename = 'tmp/job_features_train_'+ quarter + '.csv'
        with open(job_vector_filename, 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            job_postings_generator = job_postings(s3_conn, quarter)
            corpus_generator = GensimCorpusCreator().array_corpora(job_postings_generator)
            vectorized_job_generator = Doc2Vectorizer(model_name='gensim_doc2vec',
                                                      path='skills-private/model_cache/',
                                                      s3_conn=s3_conn).vectorize(corpus_generator)
            for vector in vectorized_job_generator:
                writer.writerow(vector)
        logging.info('Done vecotrizing job postings to %s', job_vector_filename)

JobVectorizeOperator(task_id='job_vectorize', dag=dag)


