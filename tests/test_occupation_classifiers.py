from skills_ml.algorithms.occupation_classifiers.classifiers import NearestNeighbors, Classifier
from gensim.similarities.index import AnnoyIndexer
from skills_utils.s3 import upload
import gensim
import os
from moto import mock_s3
from mock import patch
import boto
import tempfile

import json

import pytest


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
graduate of an approved school of practical nursing required,29,29-2061.00"""

def get_corpus(num):
    lines = [docs]*num
    for line in lines:
        yield line

class FakeCorpusGenerator(object):
    def __init__(self , num=5):
        self.num = num
        self.lookup = {}
    def __iter__(self):
        k = 1
        corpus_memory_friendly = get_corpus(num=100)
        for data in corpus_memory_friendly:
            data = gensim.utils.to_unicode(data).split(',')
            words = data[0].split()
            label = [str(k)]
            self.lookup[str(k)] = data[2]
            yield gensim.models.doc2vec.TaggedDocument(words, label)
            k += 1

@mock_s3
def test_occupation_classifier():
    s3_conn = boto.connect_s3()

    bucket_name = 'fake-bucket'
    bucket = s3_conn.create_bucket(bucket_name)

    model_id = 'test_0606'
    model_type = 'gensim_doc2vec_'
    model_name = model_type + model_id
    s3_prefix = 'fake-bucket/cache/'

    classifier_id = 'ann_0614'
    classifier_name =  classifier_id + '.index'

    fake_corpus_train = FakeCorpusGenerator(num=10)

    model = gensim.models.Doc2Vec(size=500, min_count=1, iter=5, window=4)

    with tempfile.TemporaryDirectory() as td:
        model.build_vocab(fake_corpus_train)
        model.train(fake_corpus_train)
        model.save(os.path.join(td, model_name))
        upload(s3_conn, os.path.join(td, model_name), os.path.join(s3_prefix, model_id))

    with tempfile.TemporaryDirectory() as td:
        lookup = fake_corpus_train.lookup
        lookup_name = 'lookup_' + model_id + '.json'
        with open(os.path.join(td, lookup_name), 'w') as handle:
            json.dump(lookup, handle)
        upload(s3_conn, os.path.join(td, lookup_name), os.path.join(s3_prefix, model_id))

    nn_classifier = NearestNeighbors(
        model_id=model_id,
        s3_path=s3_prefix,
        s3_conn=s3_conn,
        )


    model.init_sims()
    ann_index = AnnoyIndexer(model, 10)
    ann_classifier = NearestNeighbors(
        model_id=model_id,
        s3_path=s3_prefix,
        s3_conn=s3_conn,
        )
    ann_classifier.indexer = ann_index
    clf_top = Classifier(
        classifier_id=classifier_id,
        s3_conn=s3_conn,
        s3_path=s3_prefix,
        classifier=ann_classifier,
        classify_kwargs={'mode': 'top'}
    )
    clf_common = Classifier(
        classifier_id=classifier_id,
        s3_conn=s3_conn,
        s3_path=s3_prefix,
        classifier=ann_classifier,
        classify_kwargs={'mode': 'common'}
    )


    assert nn_classifier.model_name == model_name
    assert nn_classifier.lookup_name == lookup_name
    assert nn_classifier.indexer != clf_top.classifier.indexer
    assert nn_classifier.predict_soc(docs, 'top')[0] == clf_top.classify(docs)[0]
    assert nn_classifier.predict_soc(docs, 'common')[0] == clf_common.classify(docs)[0]

