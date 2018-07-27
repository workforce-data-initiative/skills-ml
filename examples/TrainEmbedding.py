import sys
sys.path.append('../')

import boto
s3_conn = boto.connect_s3()

import multiprocessing
cores = multiprocessing.cpu_count()

import pandas as pd
from skills_utils.time import datetime_to_quarter

from skills_ml.job_postings.common_schema import JobPostingGenerator
from skills_ml.job_postings.corpora import Doc2VecGensimCorpusCreator, Word2VecGensimCorpusCreator

from skills_ml.algorithms.embedding.train import EmbeddingTrainer


def get_time_range(start='2011-01-01', freq='Q', periods=24):

    return list(map(lambda x: datetime_to_quarter(x), pd.date_range(start=start, freq=freq, periods=periods)))

if __name__ == '__main__':
    time_range = get_time_range(start='2011-01-01', freq='Q', periods=1)
    job_postings_generator = JobPostingGenerator(s3_conn=s3_conn, quarters=time_range, s3_path='open-skills-private/job_postings_common', source="all")
    corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator)
    trainer = EmbeddingTrainer(s3_conn=s3_conn,
                               corpus_generator = corpus_generator,
                               model_s3_path='open-skills-private/model_cache/embedding/',
                               batch_size=4000,
                               model_type='word2vec')

    # The train method takes whatever arugments gensim.models.word2vec.Word2Vec or gensim.model.doc2vec.Doc2Vec has
    trainer.train(size=100, iter=4, window=8, workers=cores)

