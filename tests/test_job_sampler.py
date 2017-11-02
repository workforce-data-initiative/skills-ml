from skills_ml.algorithms.sampling.jobs import JobSampler
from skills_ml.algorithms.corpus_creators.basic import CorpusCreator
import gensim
from collections import Counter
import random
import numpy as np
import json

np.random.seed(42)
random.seed(42)

doc = {
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



class FakeCorpusGenerator(CorpusCreator):
    def __init__(self , num=5, occ_num=10):
        self.num = num
        self.lookup = {}
        self.occ_num = occ_num

    def get_corpus(self):
        occ = [2*i+11 for i in range(self.occ_num)]
        docs = [json.dumps(doc)]*self.num
        for d in docs:
            d = json.loads(d)
            d['onet_soc_code'] = str(random.choice(occ)) + d['onet_soc_code'][2:]
            yield d

    def __iter__(self):
         corpus_memory_friendly = self.get_corpus()
         for data in corpus_memory_friendly:
            yield data


def test_job_sampler_without_weighting():
    num = 1000
    occ_num = 10
    sample_size = 10
    num_loops = 200

    fake_corpus_train = FakeCorpusGenerator(num, occ_num)
    js = JobSampler(job_posting_generator=fake_corpus_train)

    result = []
    for i in range(num_loops):
        result.extend(list(map(lambda x: x['onet_soc_code'][:2], js.sample(sample_size))))

    counts = dict(Counter(result))

    assert np.mean(np.array(list(counts.values()))) == num_loops * sample_size / occ_num

def test_job_sampler_with_weighting():
    num = 1000
    occ_num = 2
    sample_size = 100
    num_loops = 200
    weights = {'11': 1, '13': 2}
    ratio = weights['13'] / weights['11']

    fake_corpus_train = FakeCorpusGenerator(num, occ_num)
    js = JobSampler(job_posting_generator=fake_corpus_train, weights=weights)

    result = []
    for i in range(num_loops):
        r = list(map(lambda x: x[1][:2], js.sample(sample_size)))
        counts = dict(Counter(r))
        result.append(counts['13'] / counts['11'])

    hist = np.histogram(result)

    # Check if the ratio of the weights (this case is 2.0) falls into the interval with maximum counts
    # in the histogram as we expect after looping for 200 times
    assert ratio >= hist[1][np.argmax(hist[0])] and ratio <= hist[1][np.argmax(hist[0]) + 1]
