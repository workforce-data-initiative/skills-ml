"""Index job titles master table

Indexes all job title and occupation pairs the API knows about
"""

import csv
import io
from datetime import datetime

from airflow import DAG
from airflow.operators import BaseOperator
from airflow.hooks import S3Hook

from config import config
from utils.es import basic_client
from utils.s3 import split_s3_path

from algorithms.elasticsearch_indexers.job_titles_master import JobTitlesMasterIndexer

default_args = {
    'depends_on_past': False,
    'start_date': datetime.today(),
}

dag = DAG(
    'index_job_titles_master',
    schedule_interval=None,
    default_args=default_args
)


class IndexJobTitlesMasterOperator(BaseOperator):
    def execute(self, context):
        conn = S3Hook()
        input_bucket, input_prefix = split_s3_path(config['output_tables']['s3_path'])
        key = conn.get_key(
            '{}/job_titles_master_table.tsv'.format(input_prefix),
            bucket_name=input_bucket
        )
        text = key.get_contents_as_string().decode('utf-8')
        reader = csv.DictReader(io.StringIO(text), delimiter='\t')
        JobTitlesMasterIndexer(
            s3_conn=conn.get_conn(),
            es_client=basic_client(),
            job_title_generator=reader
        ).replace()

IndexJobTitlesMasterOperator(task_id='job_titles_master', dag=dag)
