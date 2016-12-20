from airflow import DAG
from airflow.hooks import S3Hook
from airflow.operators import BaseOperator

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



