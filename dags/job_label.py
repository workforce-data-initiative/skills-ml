import csv
import logging
import os
from datetime import datetime
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

from algorithms.corpus_creators.basic import JobCategoryCorpusCreator

# some DAG args, please tweak for sanity
default_args = {
    'depends_on_past': False,
    'start_date': datetime(2011, 1, 1),
}

dag = DAG(
    'job_labeler',
    schedule_interval=None,
    default_args=default_args
)

class JobLabelOperator(BaseOperator):
    def execute(self, context):
        s3_conn = S3Hook().get_conn()
        quarter = datetime_to_quarter(context['execution_date'])
        job_label_filename = 'tmp/job_label_train_'+quarter+'.csv'
        with open(job_label_filename, 'w') as outfile:
            writer = csv.writer(outfile, delimiter='\n')
            job_postings_generator = job_postings(s3_conn, quarter)
            corpus_generator = JobCategoryCorpusCreator().label_corpora(job_postings_generator)
            for label in corpus_generator:
                writer.writerow(label)
        logging.info('Done labeling job categories to %s', job_label_filename)

JobLabelOperator(task_id='job_labeling', dag=dag)
