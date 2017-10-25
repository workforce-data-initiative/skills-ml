from skills_ml.algorithms.job_sampler.sampler import JobSampler
import gensim
from collections import Counter
import random
import numpy as np

np.random.seed(42)
random.seed(42)

docs = """licensed practical nurse licensed practical and licensed
vocational nurses licensed practical nurse department family
birthing center schedule part time shift hr day night rotation
hours hrs pp wknd rot holidays minimum salary minimum requisition
number job details provides direct nursing care for individual
patients undergoing cesarean section under the direction of the
surgeon also is involved with assisting with vaginal deliveries
recovery and transferring of newly delivered patient and their
families under the direction of the registered nurse to achieve
the hospital mission of competent christian holistic care patients
cared for include childbearing women and newborn infants the licensed
practical nurse can be responsible for newborn testing such as hearing
screening and car seat testing implements and abides by customer
service standards supports and implements patient safety and other
safety practices as appropriate supports and demonstrates family centered
care principles when interacting with patients and their families and
with coworkers education graduate of an approved school of practical
nursing required experience previous lpn experience preferred special
requirements current licensure as practical nurse lpn in the state of
minnesota required current american heart association aha bls healthcare
provider card required prior to completion of unit orientation eeo aa
graduate of an approved school of practical nursing required"""



class FakeCorpusGenerator(object):
    def __init__(self , num=5, occ_num=10):
        self.num = num
        self.lookup = {}
        self.occ_num = occ_num

    def get_corpus(self):
        occ = [2*i+11 for i in range(self.occ_num)]
        lines = [docs]*self.num
        for line in lines:
            yield ",".join([line, str(random.choice(occ))])

    def __iter__(self):
        k = 1
        corpus_memory_friendly = self.get_corpus()
        for data in corpus_memory_friendly:
            data = gensim.utils.to_unicode(data).split(',')
            words = data[0].split()
            label = [str(k)]
            self.lookup[str(k)] = data[1]
            yield gensim.models.doc2vec.TaggedDocument(words, label)
            k += 1


def test_job_sampler_without_weighting():
    num = 1000
    occ_num = 10
    sample_size = 10
    num_loops = 200

    fake_corpus_train = FakeCorpusGenerator(num, occ_num)
    js = JobSampler(fake_corpus_train, fake_corpus_train.lookup)

    result = []
    for i in range(num_loops):
        result.extend(list(map(lambda x: x[1],js.sample(sample_size))))

    counts = dict(Counter(result))

    assert np.mean(np.array(list(counts.values()))) == num_loops * sample_size / occ_num

def test_job_sampler_with_weighting():
    num = 1000
    occ_num = 2
    sample_size = 100
    num_loops = 200
    weights = {'11':1, '13':2}
    ratio = weights['13'] / weights['11']

    fake_corpus_train = FakeCorpusGenerator(num, occ_num)
    js = JobSampler(fake_corpus_train, fake_corpus_train.lookup, weights)

    result = []
    for i in range(num_loops):
        r = list(map(lambda x: x[1],js.sample(sample_size)))
        counts = dict(Counter(r))
        result.append(counts['13'] / counts['11'])

    hist = np.histogram(result)
    assert ratio >= hist[1][np.argmax(hist[0])] and ratio <= hist[1][np.argmax(hist[0]) + 1]
