from algorithms.job_vectorizers.doc2vec_vectorizer import Doc2Vectorizer
from algorithms.corpus_creators.basic import GensimCorpusCreator
from airflow.hooks import S3Hook
import gensim
import logging
import boto
import os
from moto import mock_s3
from tempfile import NamedTemporaryFile
from mock import MagicMock, patch

class FakeCorpusGenerator(object):
    def __init__(self, num, infer=False):
        self.num = num
        self.corpus = 'this is a job description with words for testing'
        self.tag = 'tag1'
        self.infer = infer
    def __iter__(self):
        if not self.infer:
            for i in range(self.num):
                yield gensim.models.doc2vec.TaggedDocument(self.corpus.split(), [self.tag])
        else:
            for i in range(self.num):
                yield self.corpus.split()

@mock_s3
@patch('algorithms.job_vectorizers.doc2vec_vectorizer.load2tmp')
def test_job_vectorizer(load_mock):
    model_name = 'test_doc2vec'
    s3_prefix = 'fake-bucket/cache/'
    fake_corpus_train = FakeCorpusGenerator(num=100)
    model = gensim.models.Doc2Vec(size=5, min_count=1, iter=5, window=4)
    model.build_vocab(fake_corpus_train)
    model.train(fake_corpus_train)

    expected_cache_path = 'tmp/{}'.format(model_name)

    def side_effect(s3_conn, filepath, s3_path):
        logging.warning('in side effect')
        assert filepath == expected_cache_path
        assert s3_path == '{}{}'.format(s3_prefix, model_name)
        model.save(filepath)

    load_mock.side_effect = side_effect
    fake_corpus_train_infer = FakeCorpusGenerator(num=100, infer=True)
    vectorized_job_generator = Doc2Vectorizer(model_name=model_name, path=s3_prefix).vectorize(fake_corpus_train_infer)
    assert len(model.vocab.keys()) == 9
    assert vectorized_job_generator.__next__().shape[0] == 5
    if os.path.exists(expected_cache_path):
        logging.warning('removing cache')
        os.unlink(expected_cache_path)

    assert False
