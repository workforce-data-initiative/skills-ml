"""Index job postings

Take job postings from the common schema and index them along with
likely normalized job titles
"""

from airflow import DAG
from airflow.operators import BaseOperator
from airflow.hooks import S3Hook

from utils.es import basic_client
from algorithms.elasticsearch_indexers.normalize_topn import NormalizeTopNIndexer
from utils.airflow import datetime_to_quarter
from datetime import datetime
from datasets import job_postings

# some DAG args, please tweak for sanity
default_args = {
    'depends_on_past': False,
    'start_date': datetime.today(),
}

dag = DAG(
    'normalize_elasticsearch',
    schedule_interval=None,
    default_args=default_args
)


class IndexQuarterOperator(BaseOperator):
    def execute(self, context):
        conn = S3Hook().get_conn()
        quarter = datetime_to_quarter(context['execution_date'])
        NormalizeTopNIndexer(
            quarter=quarter,
            job_postings_generator=job_postings,
            s3_conn=conn,
            es_client=basic_client()
        ).append()

common_schema_quarterly = IndexQuarterOperator(task_id='common_schema_quarterly', dag=dag)
