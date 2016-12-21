import csv
import logging
import os
from datetime import datetime
import json

from airflow import DAG
from airflow.hooks import S3Hook
from airflow.operators import BaseOperator

from utils.airflow import datetime_to_quarter
from utils.s3 import upload
from utils.hash import md5
from config import config
from datasets import job_postings

from algorithms.corpus_creators.basic import SimpleCorpusCreator
from algorithms.job_vectorizers.doc2vec_vectorizer import Doc2Vectorizer

# some DAG args, please tweak for sanity
default_args = {
    'depends_on_past': False,
    'start_date': datetime(2010, 1, 1),
}

dag = DAG(
    'corpora_labeler',
    schedule_interval=None,
    default_args=default_args
)

class JobVectorizeOperator(BaseOperator):
    def execute(self, context):
        s3_conn = S3Hook().get_conn()
        quarter = datetime_to_quarter(context['execution_date'])
        job_postings_generator = job_postings(conn, quarter)
        corpus_generator = SimpleCorpusCreator().raw_corpora(job_postings_generator)
        vectorized_job_generator = Doc2Vectorizer(model_name='gensim_doc2vec',
                                                  path='skills-private/model_cache/',
                                                  s3_conn=s3_conn).vectorize(corpus_generator)

JobVectorizeOperator(task_id='job_vectorize', dag=dag)


