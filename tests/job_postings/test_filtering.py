from functools import partial

from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.job_postings.filtering import JobPostingFilterer, soc_major_group_filter

sample_documents = [
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
    def __iter__(self):
        for document in sample_documents:
            yield document

    @property
    def metadata():
        return {'job postings': {'purpose': 'unit testing'}}


def test_JobPostingFilterer_soc_major_group():
    job_postings = FakeJobPostingGenerator()
    # Test for using pre-defined major group filter
    filtered_postings = JobPostingFilterer(
        job_postings,
        [soc_major_group_filter(major_groups=['41'])]
    )
    assert len(list(filtered_postings)) == 1


def test_JobPostingFilterer_filterfunc():
    job_postings = FakeJobPostingGenerator()
    # Test for using self-defined filter function
    def filter_func(document):
        if document['onet_soc_code']:
            if document['onet_soc_code'][:2] in ['23', '33']:
                return document

    filtered_postings = JobPostingFilterer(job_postings, filter_funcs=[filter_func])
    assert len(list(filtered_postings)) == 1
