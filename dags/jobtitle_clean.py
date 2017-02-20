import csv
import logging
import os
from datetime import datetime
import pandas as pd

from airflow import DAG
from airflow.hooks import S3Hook
from airflow.operators import BaseOperator

from utils.airflow import datetime_to_quarter
from utils.s3 import upload
from utils.nlp import NLPTransforms
from datasets import job_postings
from algorithms.jobtitle_clean.clean import JobTitleStringClean
from config import config

default_args = {
    'depends_on_past': False,
    'start_date': datetime(2011, 1, 1),
}

dag = DAG(
    'jobtitle_cleaner',
    schedule_interval=None,
    default_args=default_args
)

output_folder = config.get('output_folder', 'output')
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

class JobTitleCleanOperator(BaseOperator):
    def execute(self, context):
        s3_conn = S3Hook().get_conn()
        quarter = datetime_to_quarter(context['execution_date'])

        cleaned_count_filename = '{}/cleaned_geo_title_count_{}_new.csv'.format(
            output_folder,
            quarter
        )
        cleaned_rollup_filename = '{}/cleaned_title_count_{}.csv'.format(
            output_folder,
            quarter
        )

        count_filename = '{}/title_count_{}.csv'.format(
            output_folder,
            quarter
        )
        geo_count_filename = '{}/geo_title_count_{}.csv'.format(
            output_folder,
            quarter
        )

        geo_title_count_df = pd.read_csv(geo_count_filename, header=None)
        geo_title_count_df.columns = ['geo', 'title', 'count']
        cleaned_geo_title_count_df = JobTitleStringClean().clean(geo_title_count_df)

        title_count_df = pd.read_csv(count_filename)
        title_count_df.columns = ['title', 'count']
        cleaned_title_count_df = JobTitleStringClean().clean(title_count_df)

        with open(cleaned_count_filename, 'w') as count_file:
            clean_geo_writer = csv.writer(count_file, delimiter=',')
            for idx, row in cleaned_geo_title_count_df.iterrows():
                clean_geo_writer.writerow([row['geo'], row['title'], row['count']])

        with open(cleaned_rollup_filename, 'w') as count_file:
            clean_writer = csv.writer(count_file, delimiter=',')
            for idx, row in cleaned_title_count_df.iterrows():
                clean_writer.writerow([row['title'], row['count']])


JobTitleCleanOperator(task_id='clean_title_count', dag=dag)






