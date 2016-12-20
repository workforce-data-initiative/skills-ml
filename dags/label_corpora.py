"""
Workflow to create labeled corpora
based on common schema job listings and available skills

"""
import csv
import logging
import os
from datetime import datetime

from airflow import DAG
from airflow.hooks import S3Hook
from airflow.operators import BaseOperator

from config import config
from datasets import job_postings
from utils.airflow import datetime_to_quarter
from utils.s3 import upload
from utils.hash import md5

from algorithms.corpus_creators.basic import SimpleCorpusCreator
from algorithms.skill_taggers.simple import SimpleSkillTagger

default_args = {
    'depends_on_past': False,
    'start_date': datetime(2010, 1, 1),
}

dag = DAG(
    'corpora_labeler',
    schedule_interval='0 0 1 */3 *',
    default_args=default_args
)

output_folder = config.get('output_folder', 'output')
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
skills_filename = '{}/skills_master_table.tsv'.format(output_folder)


class SkillTagOperator(BaseOperator):
    def execute(self, context):
        conn = S3Hook().get_conn()
        quarter = datetime_to_quarter(context['execution_date'])
        labeled_filename = 'labeled_corpora_a'
        with open(labeled_filename, 'w') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            job_postings_generator = job_postings(conn, quarter)
            corpus_generator = SimpleCorpusCreator()\
                .raw_corpora(job_postings_generator)
            tagged_document_generator = \
                SimpleSkillTagger(
                    skills_filename=skills_filename,
                    hash_function=md5
                ).tagged_documents(corpus_generator)
            for document in tagged_document_generator:
                writer.writerow([document])
        logging.info('Done tagging skills to %s', labeled_filename)
        upload(
            conn,
            labeled_filename,
            '{}/{}'.format(config['labeled_postings'], quarter)
        )

SkillTagOperator(task_id='skill_tag', dag=dag)
