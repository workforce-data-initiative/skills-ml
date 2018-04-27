from skills_ml.job_postings.corpora.basic import Word2VecGensimCorpusCreator

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

class FakeJobPostingGenerator(object):
    def __iter__(self):
        yield sample_document

    @property
    def metadata(self):
        return {'entity_type': 'text corpus'}

def test_word2vec_corpus_creator():

    it = FakeJobPostingGenerator()

    corpus = [output for output in Word2VecGensimCorpusCreator(it)]
    assert len(corpus) == 1
    assert corpus[0] == 'we are looking for a person to fill this job here are some experience and requirements here are some qualifications customer service consultant entry level'.split()
