from skills_ml.algorithms.embedding.train import EmbeddingTrainer

from skills_utils.s3 import upload, list_files, download

from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.job_postings.corpora.basic import Doc2VecGensimCorpusCreator, Word2VecGensimCorpusCreator

from skills_ml.storage import S3Store, FSStore

from moto import mock_s3_deprecated, mock_s3
import mock
import boto3
import s3fs

import tempfile
import boto
import os
import json

import logging
logging.getLogger('boto').setLevel(logging.CRITICAL)

import unittest

class TestTrainEmbedding(unittest.TestCase):
    @mock_s3
    def test_embedding_trainer_doc2vec_s3(self):
        client = boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        s3_storage = S3Store(path=f"s3://fake-open-skills/model_cache/embedding")

        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']

        job_postings_generator = JobPostingCollectionSample(num_records=2)
        corpus_generator = Doc2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
        trainer = EmbeddingTrainer(corpus_generator=corpus_generator, storage=s3_storage)
        trainer.train()
        # If ones want to change the path
        s3_path_to_store = os.path.join("s3://fake-open-skills/model_cache/embedding", trainer.modelname)

        trainer.storage.path = s3_path_to_store
        trainer.save_model()
        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(s3_path_to_store)]
        assert len(files) == 3

        assert files == [trainer.modelname + '.model',
                         'lookup_' + trainer.modelname + '.json',
                         'metadata_' + trainer.modelname + '.json']

        new_trainer = EmbeddingTrainer(corpus_generator=corpus_generator, storage=s3_storage)
        self.assertRaises(NotImplementedError, lambda: new_trainer.load(trainer.modelname))

    @mock_s3
    def test_embedding_trainer_word2vec_s3(self):
        client = boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        s3_storage = S3Store(path=f"s3://fake-open-skills/model_cache/embedding")

        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        job_postings_generator = JobPostingCollectionSample(num_records=2)
        corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
        trainer = EmbeddingTrainer(corpus_generator=corpus_generator, storage=s3_storage)
        trainer.train()
        s3_path_to_store = os.path.join("s3://fake-open-skills/model_cache/embedding", trainer.modelname)
        trainer.storage.path = s3_path_to_store
        trainer.save_model()
        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(s3_path_to_store)]
        assert len(files) == 2
        assert files == ['metadata_' + trainer.modelname + '.json',
                         trainer.modelname + '.model']

        new_trainer = EmbeddingTrainer(corpus_generator=corpus_generator, storage=s3_storage)
        new_trainer.load(trainer.modelname+'.model')

        assert new_trainer.metadata['embedding_trainer']['hyperparameters'] == trainer.metadata['embedding_trainer']['hyperparameters']

    @mock.patch('os.getcwd')
    def test_embedding_trainer_doc2vec_local(self, mock_getcwd):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']

        with tempfile.TemporaryDirectory() as td:
            mock_getcwd.return_value = td
            job_postings_generator = JobPostingCollectionSample(num_records=2)
            corpus_generator = Doc2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
            trainer = EmbeddingTrainer(corpus_generator=corpus_generator, storage=FSStore(td))
            trainer.train()
            trainer.save_model()

            assert set(os.listdir(os.getcwd())) == set([trainer.modelname + '.model',
                                                             'lookup_' + trainer.modelname + '.json',
                                                             'metadata_' + trainer.modelname + '.json'])

        new_trainer = EmbeddingTrainer(corpus_generator=corpus_generator)
        self.assertRaises(NotImplementedError, lambda: new_trainer.load(trainer.modelname))

    @mock.patch('os.getcwd')
    def test_embedding_trainer_word2vec_local(self, mock_getcwd):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']

        with tempfile.TemporaryDirectory() as td:
            mock_getcwd.return_value = td
            job_postings_generator = JobPostingCollectionSample(num_records=2)
            corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
            trainer = EmbeddingTrainer(corpus_generator=corpus_generator, storage=FSStore(td))
            trainer.train()
            trainer.save_model()

            assert set(os.listdir(os.getcwd())) == set([trainer.modelname + '.model',
                                                             'metadata_' + trainer.modelname + '.json'])

            new_trainer = EmbeddingTrainer(corpus_generator, storage=FSStore(td))
            new_trainer.load(trainer.modelname + '.model')
            assert new_trainer.metadata['embedding_trainer']['hyperparameters'] == trainer.metadata['embedding_trainer']['hyperparameters']
