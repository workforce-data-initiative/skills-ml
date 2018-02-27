from airflow.hooks import S3Hook
import sys
sys.path.append('../')
s3_conn = S3Hook().get_conn()

import pandas as pd
from skills_utils.time import datetime_to_quarter

from skills_ml.algorithms.occupation_classifiers.train import RepresentationTrainer

def get_time_range(start='2011-01-01', freq='Q', periods=24):

    return list(map(lambda x: datetime_to_quarter(x), pd.date_range(start=start, freq=freq, periods=periods)))

time_range = get_time_range(start='2011-01-01', freq='Q', periods=2)

trainer = RepresentationTrainer(s3_conn=s3_conn,
                                quarters=time_range,
                                jp_s3_path='open-skills-private/job_postings_common',
                                model_s3_path='open-skills-private/model_cache')

trainer.train(size=100, iter=4, window=6, workers=3)

