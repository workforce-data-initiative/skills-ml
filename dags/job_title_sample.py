import logging
import os
from datetime import datetime

from airflow import DAG
from airflow.hooks import S3Hook
from airflow.operators import BaseOperator

from skills_ml.utils.airflow import datetime_to_quarter
from skills_ml.utils.s3 import upload
from skills_ml.utils.fs import check_create_folder
from skills_ml.algorithms.file_sampler import sampler
from config import config

default_args = {
    'depends_on_past': False,
    'start_date': datetime(2011, 1, 1),
}

dag = DAG(
    'jobtitle_sampler',
    schedule_interval='0 0 1 */3 *',
    default_args=default_args
)

output_folder = config.get('output_folder', 'output')
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

SAMPLENUM = 100

class JobTitleSampleOperator(BaseOperator):
    def execute(self, context):
        s3_conn = S3Hook().get_conn()
        quarter = datetime_to_quarter(context['execution_date'])

        cleaned_count_filename = '{}/{}/{}.csv'.format(
            output_folder,
            config.get('cleaned_geo_title_count'),
            quarter
        )

        cleaned_rollup_filename = '{}/{}/{}.csv'.format(
            output_folder,
            config.get('cleaned_title_count'),
            quarter
        )

        sampled_count_filename = '{}/sampled_geo_title_count/{}.csv'.format(
            output_folder,
            quarter
        )

        sampled_rollup_filename = '{}/sampled_title_count/{}.csv'.format(
            output_folder,
            quarter
        )

        geo_sample = sampler.reservoir_sample(SAMPLENUM, cleaned_count_filename, 12)
        count_sample = sampler.reservoir_sample(SAMPLENUM, cleaned_rollup_filename, 12)

        check_create_folder(sampled_count_filename)
        with open(sampled_count_filename, 'w') as sample_file:
            for line in geo_sample:
                sample_file.write(line)

        check_create_folder(sampled_rollup_filename)
        with open(sampled_rollup_filename, 'w') as sample_file:
            for line in count_sample:
                sample_file.write(line)

        logging.info(
            'Sampled %s job title rows for %s',
            SAMPLENUM,
            quarter,
        )

JobTitleSampleOperator(task_id='jobtitle_sample', dag=dag)
