"""
Workflow to create skills, job titles, and labeled corpora based on ONET data
and common schema job listings

"""
import csv
import logging
import os
from datetime import datetime

from airflow import DAG
from airflow.hooks import S3Hook
from airflow.operators import BaseOperator

from config import config
from datasets import job_postings, OnetCache
from utils.airflow import datetime_to_quarter
from utils.s3 import upload
from utils.hash import md5

from algorithms.corpus_creators.basic import SimpleCorpusCreator
from algorithms.skill_taggers.simple import SimpleSkillTagger
from algorithms.skill_extractors.onet_ksas import OnetSkillExtractor
from algorithms.skill_importance_extractors.onet import OnetSkillImportanceExtractor
from algorithms.title_extractors.onet import OnetTitleExtractor

# some DAG args, please tweak for sanity
default_args = {
    'depends_on_past': False,
    'start_date': datetime.today(),
}

dag = DAG(
    'simple_machine_learning',
    schedule_interval=None,
    default_args=default_args
)

output_folder = config.get('output_folder', 'output')
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
skills_filename = '{}/skills_master_table.tsv'.format(output_folder)
titles_filename = '{}/job_titles_master_table.tsv'.format(output_folder)
skill_importance_filename = '{}/ksas_importance.tsv'.format(output_folder)


class SkillExtractOperator(BaseOperator):
    def execute(self, context):
        conn = S3Hook().get_conn()
        skill_extractor = OnetSkillExtractor(
            onet_source=OnetCache(conn),
            output_filename=skills_filename,
            hash_function=md5
        )
        skill_extractor.run()
        upload(conn, skills_filename, config['output_tables']['s3_path'])


class TitleExtractOperator(BaseOperator):
    def execute(self, context):
        conn = S3Hook().get_conn()
        title_extractor = OnetTitleExtractor(
            onet_source=OnetCache(conn),
            output_filename=titles_filename,
            hash_function=md5
        )
        title_extractor.run()
        upload(conn, titles_filename, config['output_tables']['s3_path'])


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

class SkillImportanceOperator(BaseOperator):
    def execute(self, context):
        conn = S3Hook().get_conn()
        skill_extractor = OnetSkillImportanceExtractor(
            onet_source=OnetCache(conn),
            output_filename=skill_importance_filename,
            hash_function=md5
        )
        skill_extractor.run()
        upload(conn, skill_importance_filename, config['output_tables']['s3_path'])

skills = SkillExtractOperator(task_id='skill_extract', dag=dag)
titles = TitleExtractOperator(task_id='title_extract', dag=dag)
skill_tagger = SkillTagOperator(task_id='skill_tag', dag=dag)
skill_importance = SkillImportanceOperator(task_id='skill_importance', dag=dag)

skill_tagger.set_upstream(skills)
