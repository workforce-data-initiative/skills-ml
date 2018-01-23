from skills_ml.algorithms.occupation_classifiers.train import RepresentationTrainer

from skills_utils.s3 import upload, list_files, download

from moto import mock_s3_deprecated
import tempfile
import boto
import os
import json

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
def test_representation_trainer():
    s3_conn = boto.connect_s3()
    bucket_name = 'fake-jb-bucket'
    bucket = s3_conn.create_bucket(bucket_name)

    job_posting_name = 'FAKE_jobposting'
    s3_prefix_jb = 'fake-jb-bucket/job_postings'
    s3_prefix_model = 'fake-jb-bucket/model_cache'
    quarters = '2011Q1'

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, job_posting_name), 'w') as handle:
            json.dump(sample_document, handle)
        upload(s3_conn, os.path.join(td, job_posting_name), os.path.join(s3_prefix_jb, quarters))


    trainer = RepresentationTrainer(s3_conn, ['2011Q1'], s3_prefix_jb, s3_prefix_model)
    trainer.train()
    files = list_files(s3_conn, s3_prefix_model)
    assert len(files) == 3


    assert files == ['doc2vec_' + trainer.training_time + '.model',
                     'lookup_' + trainer.training_time + '.json',
                     'metadata_' + trainer.training_time + '.json']
