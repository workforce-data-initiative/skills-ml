import sys
sys.path.append('../')
#from airflow.hooks import S3Hook
#s3_conn = S3Hook().get_conn()

import boto
s3_conn = boto.connect_s3()
import multiprocessing
cores = multiprocessing.cpu_count()

import pandas as pd
from skills_utils.time import datetime_to_quarter

from skills_ml.algorithms.embedding.train import EmbeddingTrainer

def get_time_range(start='2011-01-01', freq='Q', periods=24):

    return list(map(lambda x: datetime_to_quarter(x), pd.date_range(start=start, freq=freq, periods=periods)))

if __name__ == '__main__':
    time_range = get_time_range(start='2011-01-01', freq='Q', periods=24)

    trainer = EmbeddingTrainer(s3_conn=s3_conn,
                               quarters=time_range,
                               source='nlx',
                               jp_s3_path='open-skills-private/job_postings_common',
                               model_s3_path='open-skills-private/model_cache/embedding/',
                               batch_size=4000,
                               model_type='word2vec')

    # The train method takes whatever arugments gensim.models.word2vec.Word2Vec or gensim.model.doc2vec.Doc2Vec has
    trainer.train(size=100, iter=4, window=8, workers=cores)

