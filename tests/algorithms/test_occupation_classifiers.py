from skills_ml.algorithms.occupation_classifiers.classifiers import CombinedClassifier, KNNDoc2VecClassifier, SocClassifier
from skills_ml.algorithms.embedding.train import EmbeddingTrainer
from skills_ml.algorithms.occupation_classifiers import SOCMajorGroup, DesignMatrix
from skills_ml.algorithms.embedding.models import Doc2VecModel, Word2VecModel, EmbeddingTransformer
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.job_postings.corpora import Word2VecGensimCorpusCreator
from skills_ml.storage import ProxyObjectWithStorage, ModelStorage, S3Store, FSStore
from skills_ml.algorithms import nlp
from skills_ml.algorithms.preprocessing import IterablePipeline

from skills_utils.s3 import upload

import gensim
from gensim.similarities.index import AnnoyIndexer

from sklearn.ensemble import RandomForestClassifier

from moto import mock_s3
from descriptors import cachedproperty
from functools import partial
import mock
import boto3
import s3fs
import tempfile
import os
import unittest
import json



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
    def __init__(self , num=25):
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

class TestCombinedClassifier(unittest.TestCase):
    def basic_filter(self, doc):
         if self.major_group.filter_func(doc):
             return doc
         else:
             return None

    @property
    def pipe_x(self):
        document_schema_fields = ['description', 'experienceRequirements', 'qualifications', 'skills']
        pipe_x = IterablePipeline(
                self.basic_filter,
                partial(nlp.fields_join, document_schema_fields=document_schema_fields),
                nlp.clean_str,
                nlp.word_tokenize,
                partial(nlp.vectorize, embedding_model=Word2VecModel(size=10))
        )
        return pipe_x

    @property
    def pipe_y(self):
        pipe_y = IterablePipeline(
                self.basic_filter,
                self.major_group.transformer
        )
        return pipe_y

    @cachedproperty
    def major_group(self):
        return SOCMajorGroup()

    @mock.patch('os.getcwd')
    def test_combined_cls_local(self, mock_getcwd):
        with tempfile.TemporaryDirectory() as td:
            mock_getcwd.return_value = td
            model_storage = ModelStorage(FSStore(td))
            jobpostings = JobPostingCollectionSample()
            corpus_generator = Word2VecGensimCorpusCreator(jobpostings, raw=True)
            w2v = Word2VecModel(size=10, min_count=0, alpha=0.025, min_alpha=0.025)
            trainer = EmbeddingTrainer(w2v, model_storage=model_storage)
            trainer.train(corpus_generator, True)

            matrix = DesignMatrix(jobpostings, self.major_group, self.pipe_x, self.pipe_y)
            matrix.build()

            X = matrix.X
            rf = ProxyObjectWithStorage(RandomForestClassifier(), None, None, matrix.target_variable)
            rf.fit(X, matrix.y)

            proxy_rf = ProxyObjectWithStorage(rf, None, None, matrix.target_variable)
            # Remove the last step in the pipe_x
            # the input of predict_soc should be tokenized words
            new_pipe_x = self.pipe_x
            new_pipe_x.generators.pop()

            new_matrix = DesignMatrix(JobPostingCollectionSample(), self.major_group, new_pipe_x)
            new_matrix.build()
            ccls = CombinedClassifier(w2v, rf)
            assert len(ccls.predict_soc([new_matrix.X[0]])[0]) == 2

class TestKNNDoc2VecClassifier(unittest.TestCase):
    @mock.patch('os.getcwd')
    def test_knn_doc2vec_cls_local(self, mock_getcwd):
        with tempfile.TemporaryDirectory() as td:
            mock_getcwd.return_value = td
            model_storage = ModelStorage(FSStore(td))
            corpus_generator = FakeCorpusGenerator()
            d2v = Doc2VecModel(size=10, min_count=1, dm=0, alpha=0.025, min_alpha=0.025)
            trainer = EmbeddingTrainer(d2v, model_storage=model_storage)
            trainer.train(corpus_generator, True)

            # KNNDoc2VecClassifier only supports doc2vec now
            self.assertRaises(NotImplementedError, lambda: KNNDoc2VecClassifier(Word2VecModel()))

            doc = docs.split(',')[0].split()

            knn = KNNDoc2VecClassifier(embedding_model=d2v, k=0)
            self.assertRaises(ValueError, lambda: knn.predict_soc([doc]))

            knn = KNNDoc2VecClassifier(embedding_model=d2v, k=1)
            soc_cls = SocClassifier(knn)

            assert knn.predict_soc([doc])[0][0] == soc_cls.predict_soc([doc])[0][0]

            # Build Annoy index
            knn.build_ann_indexer(num_trees=5)
            assert isinstance(knn.indexer, AnnoyIndexer)

            # Save
            model_storage.save_model(knn, knn.model_name)
            assert set(os.listdir(os.getcwd())) == set([knn.model_name])
            assert isinstance(knn.indexer, AnnoyIndexer)

            # Load
            new_knn = model_storage.load_model(knn.model_name)
            assert new_knn.model_name ==  knn.model_name
            assert new_knn.predict_soc([doc])[0][0] == '29-2061.00'

            # Have to re-build the index whenever ones load the knn model to the memory
            assert new_knn.indexer == None

    @mock_s3
    def test_knn_doc2vec_cls_s3(self):
        client = boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        s3_path = f"s3://fake-open-skills/model_cache/soc_classifiers"
        s3_storage = S3Store(path=s3_path)
        model_storage = ModelStorage(s3_storage)
        corpus_generator = FakeCorpusGenerator()

        # Embedding has no lookup_dict
        d2v = Doc2VecModel(size=10, min_count=1, dm=0, alpha=0.025, min_alpha=0.025)
        trainer = EmbeddingTrainer(d2v, model_storage=model_storage)
        trainer.train(corpus_generator, lookup=False)

        self.assertRaises(ValueError, lambda: KNNDoc2VecClassifier(embedding_model=d2v))

        d2v = Doc2VecModel(size=10, min_count=1, dm=0, alpha=0.025, min_alpha=0.025)
        trainer = EmbeddingTrainer(d2v, model_storage=model_storage)
        trainer.train(corpus_generator, lookup=True)

        # KNNDoc2VecClassifier only supports doc2vec now
        self.assertRaises(NotImplementedError, lambda: KNNDoc2VecClassifier(Word2VecModel()))

        doc = docs.split(',')[0].split()

        knn = KNNDoc2VecClassifier(embedding_model=d2v, k=0)
        self.assertRaises(ValueError, lambda: knn.predict_soc([doc]))

        knn = KNNDoc2VecClassifier(embedding_model=d2v, k=10)
        soc_cls = SocClassifier(knn)

        assert knn.predict_soc([doc])[0][0] == soc_cls.predict_soc([doc])[0][0]


        # Build Annoy index
        knn.build_ann_indexer(num_trees=5)
        assert isinstance(knn.indexer, AnnoyIndexer)

        # Save
        s3 = s3fs.S3FileSystem()
        model_storage.save_model(knn, knn.model_name)
        files = [f.split('/')[-1] for f in s3.ls(s3_path)]
        assert set(files) == set([knn.model_name])

        # Load
        new_knn = model_storage.load_model(knn.model_name)
        assert new_knn.model_name ==  knn.model_name
        assert new_knn.predict_soc([doc])[0][0] == '29-2061.00'

        # Have to re-build the index whenever ones load the knn model to the memory
        assert new_knn.indexer == None
