from skills_ml.algorithms.embedding.train import EmbeddingTrainer

from skills_utils.s3 import upload, list_files, download

from skills_ml.job_postings.common_schema import JobPostingGenerator
from skills_ml.job_postings.corpora.basic import Doc2VecGensimCorpusCreator, Word2VecGensimCorpusCreator

from moto import mock_s3_deprecated
import tempfile
import boto
import os
import json

import logging
logging.getLogger('boto').setLevel(logging.CRITICAL)


sample_document = {
    "incentiveCompensation": "",
    "experienceRequirements": "Here are some experience and requirements",
    "baseSalary": {
        "maxValue": 0.0,
        "@type": "MonetaryAmount",
        "minValue": 0.0
    },
    "description": "We are looking for a person to fill this job",
    "title": "Bilingual (Italian) Customer Service Rep (Work from Home)",
    "employmentType": "Full-Time",
    "industry": "Call Center / SSO / BPO, Consulting, Sales - Marketing",
    "occupationalCategory": "",
    "onet_soc_code": "41-1011.00",
    "qualifications": "Here are some qualifications",
    "educationRequirements": "Not Specified",
    "skills": "Customer Service, Consultant, Entry Level",
    "validThrough": "2014-01-02T00:00:00",
    "jobLocation": {
        "@type": "Place",
        "address": {
            "addressLocality": "Salisbury",
            "addressRegion": "PA",
            "@type": "PostalAddress"
        }
    },
    "@context": "http://schema.org",
    "alternateName": "Customer Service Representative",
    "datePosted": "2013-05-12",
    "@type": "JobPosting"
}

@mock_s3_deprecated
def test_embedding_trainer():
    s3_conn = boto.connect_s3()
    bucket_name = 'fake-jb-bucket'
    bucket = s3_conn.create_bucket(bucket_name)

    job_posting_name = 'FAKE_jobposting'
    s3_prefix_jb = 'fake-jb-bucket/job_postings'
    s3_prefix_model = 'fake-jb-bucket/model_cache/embedding/'
    quarters = '2011Q1'

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, job_posting_name), 'w') as handle:
            json.dump(sample_document, handle)
        upload(s3_conn, os.path.join(td, job_posting_name), os.path.join(s3_prefix_jb, quarters))


    # Doc2Vec
    job_postings_generator = JobPostingGenerator(s3_conn=s3_conn, quarters=['2011Q1'], s3_path=s3_prefix_jb, source="all")
    corpus_generator = Doc2VecGensimCorpusCreator(job_postings_generator)
    trainer = EmbeddingTrainer(corpus_generator=corpus_generator, s3_conn=s3_conn, model_s3_path=s3_prefix_model, model_type='doc2vec')
    trainer.train()
    files = list_files(s3_conn, os.path.join(s3_prefix_model, 'doc2vec_gensim_' + trainer.training_time))
    assert len(files) == 3

    assert files == ['doc2vec_gensim_' + trainer.training_time + '.model',
                     'lookup_doc2vec_gensim_' + trainer.training_time + '.json',
                     'metadata_doc2vec_gensim_' + trainer.training_time + '.json']

    with tempfile.TemporaryDirectory() as td:
        trainer.save_model(td)
        assert set(os.listdir(td)) == set(['doc2vec_gensim_' + trainer.training_time + '.model',
                                           'lookup_doc2vec_gensim_' + trainer.training_time + '.json',
                                           'metadata_doc2vec_gensim_' + trainer.training_time + '.json'])

    # Word2Vec
    job_postings_generator = JobPostingGenerator(s3_conn=s3_conn, quarters=['2011Q1'], s3_path=s3_prefix_jb, source="all")
    corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator)
    trainer = EmbeddingTrainer(corpus_generator=corpus_generator, s3_conn=s3_conn, model_s3_path=s3_prefix_model, model_type='word2vec')
    trainer.train()
    files = list_files(s3_conn, os.path.join(s3_prefix_model, 'word2vec_gensim_' + trainer.training_time))
    assert len(files) == 2
    assert files == ['metadata_word2vec_gensim_' + trainer.training_time + '.json',
                     'word2vec_gensim_' + trainer.training_time + '.model']

    job_postings_generator = JobPostingGenerator(s3_conn=s3_conn, quarters=['2011Q1'], s3_path=s3_prefix_jb, source="all")
    corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator)
    new_trainer = EmbeddingTrainer(corpus_generator=corpus_generator, s3_conn=s3_conn, model_s3_path=s3_prefix_model, model_type='word2vec')
    new_trainer.load(trainer.modelname, s3_prefix_model)
    assert new_trainer.metadata['metadata']['hyperparameters'] == trainer.metadata['metadata']['hyperparameters']
