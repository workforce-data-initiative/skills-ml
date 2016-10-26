from airflow import DAG
from airflow.operators import DummyOperator, PythonOperator
from datetime import datetime
from evaluation.query import generate_evaluators, run_evaluator

# some DAG args, please tweak for sanity
default_args = {
    'evaluation_file': 'interesting_job_titles.csv',
    'owner': 'job_title_normalizer',
    'depends_on_past': True,
    'start_date': datetime.today()
}

dag = DAG('job_title_normalizer_evaluation',
          schedule_interval=None,
          default_args=default_args)

run_this = DummyOperator(
    task_id='Root',
    dag=dag)

for class_name, options in generate_evaluators(default_args['evaluation_file']):
    task = PythonOperator(
        task_id=options['name'],
        python_callable=run_evaluator,
        op_args=[class_name],
        op_kwargs=options,
        dag=dag)
    task.set_upstream(run_this)
