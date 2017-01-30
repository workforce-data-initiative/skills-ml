import csv
import logging
import os
from datetime import datetime

from airflow import DAG
from airflow.hooks import S3Hook
from airflow.operators import BaseOperator

from utils.airflow import datetime_to_quarter
from utils.s3 import upload
from utils.nlp import NLPTransforms
from datasets import job_postings
from algorithms.aggregators.title import GeoTitleAggregator
from config import config

default_args = {
    'depends_on_past': False,
    'start_date': datetime(2011, 1, 1),
}

dag = DAG(
    'geo_title_counts',
    schedule_interval='0 0 1 */3 *',
    default_args=default_args
)

output_folder = config.get('output_folder', 'output')
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)


class GeoTitleCountOperator(BaseOperator):
    def execute(self, context):
        s3_conn = S3Hook().get_conn()
        quarter = datetime_to_quarter(context['execution_date'])
        count_filename = '{}/geo_title_count_{}.csv'.format(
            output_folder,
            quarter
        )
        rollup_filename = '{}/title_count_{}.csv'.format(
            output_folder,
            quarter
        )

        job_postings_generator = job_postings(s3_conn, quarter)
        title_cleaner = NLPTransforms().lowercase_strip_punc
        counts, title_rollup = GeoTitleAggregator(title_cleaner=title_cleaner)\
            .counts(job_postings_generator)

        total_counts = 0
        with open(count_filename, 'w') as count_file:
            count_writer = csv.writer(count_file, delimiter=',')
            for key, count in counts.items():
                geo, title = key
                total_counts += count
                count_writer.writerow([geo, title, count])

        rollup_counts = 0
        with open(rollup_filename, 'w') as rollup_file:
            rollup_writer = csv.writer(rollup_file, delimiter=',')
            for title, count in title_rollup.items():
                rollup_counts += count
                rollup_writer.writerow([title, count])

        logging.info(
            'Found %s count rows and %s title rollup rows for %s',
            total_counts,
            rollup_counts,
            quarter,
        )
        upload(s3_conn, count_filename, config['output_tables']['s3_path'])
        upload(s3_conn, rollup_filename, config['output_tables']['s3_path'])

GeoTitleCountOperator(task_id='geo_title_count', dag=dag)
