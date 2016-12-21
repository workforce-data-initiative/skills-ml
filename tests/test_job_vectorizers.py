from algorithms.job_vectorizers.doc2vec_vectorizer import Doc2Vectorizer
from airflow.hooks import S3Hook

def test_job_vectorizer():
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
    MODEL_NAME = 'gensim_doc2vec'
    PATHTOMODEL = 'skills-private/model_cache/'
    s3_conn = S3Hook().get_conn()
    job_vec = Doc2Vectorizer(model_name=MODEL_NAME, path=PATHTOMODEL, s3_conn=s3_conn).vectorize(sample_document['description'])

    assert job_vec.shape[0] == 500
