from algorithms.job_vectorizers.doc2vec_vectorizer import Doc2Vectorizer
from algorithms.corpus_creators.basic import GensimCorpusCreator
from airflow.hooks import S3Hook

from datasets import job_postings

def test_job_vectorizer():
    MODEL_NAME = 'gensim_doc2vec'
    PATHTOMODEL = 'skills-private/model_cache/'
    s3_conn = S3Hook().get_conn()

    job_postings_generator = job_postings(s3_conn, '2011Q1')
    corpus_generator = GensimCorpusCreator().array_corpora(job_postings_generator)
    #print(corpus_generator.__next__())
    assert isinstance(corpus_generator.__next__(), list)


    vectorized_job_generator = Doc2Vectorizer(model_name=MODEL_NAME,
                                              path=PATHTOMODEL,
                                              s3_conn=s3_conn).vectorize(corpus_generator)

    #print(vectorized_job_generator.__next__())
    assert vectorized_job_generator.__next__().shape[0] == 500




