import csv
import logging
from datetime import datetime

from airflow import DAG
from airflow.hooks import S3Hook
from airflow.operators import BaseOperator

from utils.airflow import datetime_to_quarter
from datasets import job_postings

from algorithms.corpus_creators.basic import GensimCorpusCreator
from algorithms.job_vectorizers.doc2vec_vectorizer import Doc2Vectorizer

# some DAG args, please tweak for sanity
default_args = {
    'depends_on_past': False,
    'start_date': datetime(2011, 1, 1),
}

dag = DAG(
    'job_vectorizer',
    schedule_interval=None,
    default_args=default_args
)

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


