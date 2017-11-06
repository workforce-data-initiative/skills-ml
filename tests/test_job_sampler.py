from skills_ml.algorithms.sampling.jobs import JobSampler
from skills_utils.common import safe_get
import gensim
from collections import Counter
import random
import numpy as np
import json
import unittest

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



class FakeCorpusGenerator(object):
    def __init__(self , num=5, occ_num=10, states=['PA', 'IL'], employment_type=['Full-Time', 'Part-Time']):
        self.num = num
        self.occ_num = occ_num
        self.states = states
        self.employment_type = employment_type

    def get_corpus(self):
        occ = [2*i+11 for i in range(self.occ_num)]
        docs = [json.dumps(doc)]*self.num
        for d in docs:
            d = json.loads(d)
            d['onet_soc_code'] = str(random.choice(occ)) + d['onet_soc_code'][2:]
            d['jobLocation']['address']['addressRegion'] = random.choice(self.states)
            d['employmentType'] = random.choice(self.employment_type)
            yield d

    def __iter__(self):
         corpus_memory_friendly = self.get_corpus()
         for data in corpus_memory_friendly:
            yield data


class JobSamplerWithoutWeightingTest(unittest.TestCase):
    num = 1000
    occ_num = 10
    sample_size = 10
    num_loops = 200
    states = ['PA', 'IL']
    employment_type=['Full-Time', 'Part-Time']
    fake_corpus_train = FakeCorpusGenerator(num, occ_num, states, employment_type)

    def test_soc(self):
        js = JobSampler(job_posting_generator=self.fake_corpus_train)

        result = []
        for i in range(self.num_loops):
            result.extend(list(map(lambda x: x[0]['onet_soc_code'], js.sample(self.sample_size))))

        counts = dict(Counter(result))
        assert np.mean(np.array(list(counts.values()))) == self.num_loops * self.sample_size / self.occ_num

    def test_state(self):
        js = JobSampler(job_posting_generator=self.fake_corpus_train, keys=['jobLocation', 'address', 'addressRegion'])

        result = []
        for i in range(self.num_loops):
            result.extend(list(map(lambda x: x[1], js.sample(self.sample_size))))

        counts = dict(Counter(result))
        assert np.mean(np.array(list(counts.values()))) == self.num_loops * self.sample_size / len(self.states)

    def test_employment_type(self):
        js = JobSampler(job_posting_generator=self.fake_corpus_train, keys='employmentType')

        result = []
        for i in range(self.num_loops):
            result.extend(list(map(lambda x: x[1], js.sample(self.sample_size))))

        counts = dict(Counter(result))
        assert np.mean(np.array(list(counts.values()))) == self.num_loops * self.sample_size / len(self.employment_type)


class JobSamplerWithWeightingTest(unittest.TestCase):
    num = 1000
    occ_num = 2
    sample_size = 100
    num_loops = 200
    weights = {'11': 1, '13': 2}
    fake_corpus_train = FakeCorpusGenerator(num, occ_num)

    def test_major_group(self):

        ratio = self.weights['13'] / self.weights['11']

        js = JobSampler(job_posting_generator=self.fake_corpus_train, weights=self.weights, major_group=True)

        result = []
        for i in range(self.num_loops):
            r = list(map(lambda x: x[1][:2], js.sample(self.sample_size)))
            counts = dict(Counter(r))
            result.append(counts['13'] / counts['11'])

        hist = np.histogram(result, bins=[0, 1, 2, 3, 4, 5])

        # Check if the ratio of the weights (this case is 2.0) falls into the interval with maximum counts
        # in the histogram as we expect after looping for 200 times
        assert ratio >= hist[1][np.argmax(hist[0])] and ratio <= hist[1][np.argmax(hist[0]) + 1]
