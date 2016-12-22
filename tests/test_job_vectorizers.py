from algorithms.job_vectorizers.doc2vec_vectorizer import Doc2Vectorizer
from algorithms.corpus_creators.basic import GensimCorpusCreator
from airflow.hooks import S3Hook
import gensim
import boto
from moto import mock_s3
from tempfile import NamedTemporaryFile

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
def test_job_vectorizer():
    MODEL_NAME = 'test_model'
    PATHTOMODEL = 'skills-private/model_cache/'
    s3_conn = boto.connect_s3()
    bucket = s3_conn.create_bucket('skills-private')

    key = boto.s3.key.Key(
        bucket=bucket,
        name='model_cache/test_model'
    )

    fake_corpus_train = FakeCorpusGenerator(num=100)
    model = gensim.models.Doc2Vec(size=500, min_count=1, iter=5, window=4)
    model.build_vocab(fake_corpus_train)
    model.train(fake_corpus_train)

    temp_model = NamedTemporaryFile()
    model.save(temp_model.name)
    temp_model.seek(0)
    key.set_contents_from_filename(temp_model.name)
    fake_corpus_train_infer = FakeCorpusGenerator(num=100, infer=True)
    vectorized_job_generator = Doc2Vectorizer(model_name=MODEL_NAME,
                                              path=PATHTOMODEL,
                                              s3_conn=s3_conn).vectorize(fake_corpus_train_infer)

    assert len(model.vocab.keys()) == 9
    assert vectorized_job_generator.__next__().shape[0] == 500





