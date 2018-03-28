from skills_ml.job_postings.corpora.basic import Doc2VecGensimCorpusCreator
import json


sample_document = [
    {
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
    },
    {
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
        "onet_soc_code": "23-1011.00",
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
    },
]

class FakeJobPostingGenerator(object):
    def __init__(self):
        self.quarters = ['2013Q2']
    def __iter__(self):
        for d in sample_document:
            yield json.dumps(d)


def test_doc2vec_corpus_creator():

    it = FakeJobPostingGenerator()

    corpus = Doc2VecGensimCorpusCreator(it)._clean(sample_document[0])
    assert corpus == 'we are looking for a person to fill this job here are some experience and requirements here are some qualifications customer service consultant entry level'

    # Test for Default
    corpus = Doc2VecGensimCorpusCreator(it)
    assert len(list(corpus)) == 2

    # Test for using pre-defined major group filter
    corpus = Doc2VecGensimCorpusCreator(it, major_groups=['41'])
    assert len(list(corpus)) == 1


    # Test for using self-defined filter function
    def filter_func(document):
        if document['onet_soc_code']:
            if document['onet_soc_code'][:2] in ['23', '33']:
                return document

    corpus = Doc2VecGensimCorpusCreator(it, filter_func=filter_func, key='onet_soc_code')
    assert len(list(corpus)) == 1
