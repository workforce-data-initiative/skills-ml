from skills_ml.algorithms.embedding.train import EmbeddingTrainer
from skills_ml.algorithms.embedding.models import Word2VecModel, Doc2VecModel
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.job_postings.corpora.basic import Doc2VecGensimCorpusCreator, Word2VecGensimCorpusCreator
from skills_ml.storage import S3Store, FSStore

from skills_utils.s3 import list_files

from moto import mock_s3
import mock
import boto3
import s3fs
import tempfile
import os
import unittest

import logging
logging.getLogger('boto').setLevel(logging.CRITICAL)


class TestTrainEmbedding(unittest.TestCase):
    @mock_s3
    def test_embedding_trainer_doc2vec_s3(self):
        client = boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        s3_path = f"s3://fake-open-skills/model_cache/embedding"
        s3_storage = S3Store(path=s3_path)

        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        job_postings_generator = JobPostingCollectionSample(num_records=30)
        corpus_generator = Doc2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
        d2v = Doc2VecModel(storage=s3_storage, size=10, min_count=3, iter=4, window=6, workers=3)

        trainer = EmbeddingTrainer(corpus_generator, d2v)
        trainer.train(lookup=True)
        trainer.save_model()

        vocab_size = len(d2v.wv.vocab.keys())
        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(s3_path)]
        assert d2v.model_name == trainer.model_name
        assert set(files) == set([trainer.model_name])
        self.assertDictEqual(trainer.lookup_dict, d2v.lookup_dict)

        # Save as different name
        d2v.save('other_name.model')

        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(s3_path)]
        assert set(files) == set([trainer.model_name, 'other_name.model'])

        # Load
        d2v_loaded = Doc2VecModel.load(s3_storage, trainer.model_name)
        self.assertDictEqual(d2v_loaded.metadata, trainer.metadata)

        # Change the store directory
        new_s3_path = "s3://fake-open-skills/model_cache/embedding/other_directory"
        trainer.save_model(S3Store(new_s3_path))
        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(new_s3_path)]
        assert set(files) == set([trainer.model_name])

    @mock_s3
    def test_embedding_trainer_word2vec_s3(self):
        client = boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        s3_path = f"s3://fake-open-skills/model_cache/embedding"
        s3_storage = S3Store(path=s3_path)

        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        job_postings_generator = JobPostingCollectionSample(num_records=30)
        corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
        w2v = Word2VecModel(storage=s3_storage, size=10, min_count=3, iter=4, window=6, workers=3)

        trainer = EmbeddingTrainer(corpus_generator, w2v)
        trainer.train()
        trainer.save_model()

        vocab_size = len(w2v.wv.vocab.keys())

        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(s3_path)]
        assert w2v.model_name == trainer.model_name
        assert set(files) == set([trainer.model_name])

        # Test online training
        job_postings_generator = JobPostingCollectionSample(num_records=50)
        corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)

        w2v_loaded = Word2VecModel.load(s3_storage, w2v.model_name)

        new_trainer = EmbeddingTrainer(corpus_generator, w2v_loaded)
        new_trainer.train()
        new_trainer.save_model()

        new_vocab_size = len(w2v_loaded.wv.vocab.keys())

        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(s3_path)]
        assert set(files) == set([new_trainer.model_name, trainer.model_name])
        assert new_trainer.metadata['embedding_trainer']['hyperparameters'] == trainer.metadata['embedding_trainer']['hyperparameters']
        assert vocab_size <= new_vocab_size

        # Save as different name
        w2v.save('other_name.model')

        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(s3_path)]
        assert set(files) == set([trainer.model_name, new_trainer.model_name, 'other_name.model'])

        # Change the store directory
        new_s3_path = "s3://fake-open-skills/model_cache/embedding/other_directory"
        new_trainer.save_model(S3Store(new_s3_path))
        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(new_s3_path)]
        assert set(files) == set([new_trainer.model_name])


    @mock.patch('os.getcwd')
    def test_embedding_trainer_doc2vec_local(self, mock_getcwd):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']

        with tempfile.TemporaryDirectory() as td:
            mock_getcwd.return_value = td
            job_postings_generator = JobPostingCollectionSample(num_records=30)
            corpus_generator = Doc2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
            d2v = Doc2VecModel(storage=FSStore(td), size=10, min_count=3, iter=4, window=6, workers=3)

            trainer = EmbeddingTrainer(corpus_generator, d2v)
            trainer.train(lookup=True)
            trainer.save_model()

            vocab_size = len(d2v.wv.vocab.keys())
            assert d2v.model_name == trainer.model_name
            assert set(os.listdir(os.getcwd())) == set([trainer.model_name])
            self.assertDictEqual(trainer.lookup_dict, d2v.lookup_dict)

            # Save as different name
            d2v.save('other_name.model')
            assert set(os.listdir(os.getcwd())) == set([trainer.model_name, 'other_name.model'])

            # Load
            d2v_loaded = Doc2VecModel.load(FSStore(td), trainer.model_name)
            self.assertDictEqual(d2v_loaded.metadata, trainer.metadata)

            # Change the store directory
            new_path = os.path.join(td, 'other_directory')
            trainer.save_model(FSStore(new_path))
            assert set(os.listdir(new_path)) == set([trainer.model_name])

    @mock.patch('os.getcwd')
    def test_embedding_trainer_word2vec_local(self, mock_getcwd):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']

        with tempfile.TemporaryDirectory() as td:
            mock_getcwd.return_value = td
            job_postings_generator = JobPostingCollectionSample(num_records=30)
            corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
            w2v = Word2VecModel(storage=FSStore(td), size=10, min_count=3, iter=4, window=6, workers=3)

            trainer = EmbeddingTrainer(corpus_generator, w2v)
            trainer.train()
            trainer.save_model()

            vocab_size = len(w2v.wv.vocab.keys())

            assert w2v.model_name == trainer.model_name
            assert set(os.listdir(os.getcwd())) == set([trainer.model_name])

            # Test Online Training
            job_postings_generator = JobPostingCollectionSample(num_records=50)
            corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)

            w2v_loaded = Word2VecModel.load(FSStore(td), w2v.model_name)

            new_trainer = EmbeddingTrainer(corpus_generator, w2v_loaded)
            new_trainer.train()
            new_trainer.save_model()

            new_vocab_size = len(w2v_loaded.wv.vocab.keys())

            assert set(os.listdir(os.getcwd())) == set([trainer.model_name, new_trainer.model_name])
            assert new_trainer.metadata['embedding_trainer']['hyperparameters'] == trainer.metadata['embedding_trainer']['hyperparameters']
            assert vocab_size <= new_vocab_size

            # Save as different name
            w2v.save('other_name.model')
            assert set(os.listdir(os.getcwd())) == set([trainer.model_name, new_trainer.model_name, 'other_name.model'])

            # Change the store directory
            new_path = os.path.join(td, 'other_directory')
            new_trainer.save_model(FSStore(new_path))
            assert set(os.listdir(new_path)) == set([new_trainer.model_name])
