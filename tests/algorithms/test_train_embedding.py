from skills_ml.algorithms.embedding.train import EmbeddingTrainer
from skills_ml.algorithms.embedding.models import Word2VecModel, Doc2VecModel, FastTextModel
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.job_postings.corpora import Doc2VecGensimCorpusCreator, Word2VecGensimCorpusCreator
from skills_ml.storage import ModelStorage, S3Store, FSStore

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
        model_storage = ModelStorage(s3_storage)

        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        job_postings_generator = JobPostingCollectionSample(num_records=30)
        corpus_generator = Doc2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
        d2v = Doc2VecModel(size=10, min_count=3, iter=4, window=6, workers=3)

        trainer = EmbeddingTrainer(d2v, model_storage=model_storage)
        trainer.train(corpus_generator, lookup=True)
        trainer.save_model()

        vocab_size = len(d2v.wv.vocab.keys())
        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(s3_path)]
        assert d2v.model_name == trainer._models[0].model_name
        assert set(files) == set([trainer._models[0].model_name])
        print(trainer.lookup_dict)
        print(d2v.lookup_dict)
        self.assertDictEqual(trainer.lookup_dict, d2v.lookup_dict)
        # Save as different name
        model_storage.save_model(d2v, 'other_name.model')

        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(s3_path)]
        assert set(files) == set([trainer._models[0].model_name, 'other_name.model'])

        # Load
        d2v_loaded = model_storage.load_model(trainer._models[0].model_name)
        assert d2v_loaded.metadata['embedding_model']['hyperparameters']['vector_size'] ==  trainer._models[0].metadata['embedding_model']['hyperparameters']['vector_size']
        # Change the store directory
        new_s3_path = "s3://fake-open-skills/model_cache/embedding/other_directory"
        trainer.save_model(S3Store(new_s3_path))
        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(new_s3_path)]
        assert set(files) == set([trainer._models[0].model_name])

    @mock_s3
    def test_embedding_trainer_word2vec_s3(self):
        client = boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        s3_path = f"s3://fake-open-skills/model_cache/embedding"
        s3_storage = S3Store(path=s3_path)
        model_storage = ModelStorage(s3_storage)

        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        job_postings_generator = JobPostingCollectionSample(num_records=30)
        corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
        w2v = Word2VecModel(size=10, min_count=3, iter=4, window=6, workers=3)

        trainer = EmbeddingTrainer(w2v, model_storage=model_storage)
        trainer.train(corpus_generator)
        trainer.save_model()

        vocab_size = len(w2v.wv.vocab.keys())

        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(s3_path)]
        assert w2v.model_name == trainer._models[0].model_name
        assert set(files) == set([trainer._models[0].model_name])

        # Test online training
        job_postings_generator = JobPostingCollectionSample(num_records=50)
        corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)

        w2v_loaded = model_storage.load_model(w2v.model_name)

        new_trainer = EmbeddingTrainer(w2v_loaded, model_storage=model_storage)
        new_trainer.train(corpus_generator)
        new_trainer.save_model()

        new_vocab_size = len(w2v_loaded.wv.vocab.keys())

        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(s3_path)]
        assert set(files) == set([new_trainer._models[0].model_name, trainer._models[0].model_name])
        assert new_trainer.metadata['embedding_trainer']['models'] != trainer.metadata['embedding_trainer']['models']
        assert vocab_size <= new_vocab_size

        # Save as different name
        model_storage.save_model(w2v, 'other_name.model')

        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(s3_path)]
        assert set(files) == set([trainer._models[0].model_name, new_trainer._models[0].model_name, 'other_name.model'])

        # Change the store directory
        new_s3_path = "s3://fake-open-skills/model_cache/embedding/other_directory"
        new_trainer.save_model(S3Store(new_s3_path))
        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(new_s3_path)]
        assert set(files) == set([new_trainer._models[0].model_name])


    @mock.patch('os.getcwd')
    def test_embedding_trainer_doc2vec_local(self, mock_getcwd):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']

        with tempfile.TemporaryDirectory() as td:
            mock_getcwd.return_value = td
            model_storage = ModelStorage(FSStore(td))

            job_postings_generator = JobPostingCollectionSample(num_records=30)
            corpus_generator = Doc2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
            d2v = Doc2VecModel(size=10, min_count=3, iter=4, window=6, workers=3)


            trainer = EmbeddingTrainer(d2v, model_storage=model_storage)
            trainer.train(corpus_generator, lookup=True)
            trainer.save_model()

            vocab_size = len(d2v.wv.vocab.keys())
            assert d2v.model_name == trainer._models[0].model_name
            assert set(os.listdir(os.getcwd())) == set([trainer._models[0].model_name])
            self.assertDictEqual(trainer.lookup_dict, d2v.lookup_dict)

            # Save as different name
            model_storage.save_model(d2v, 'other_name.model')
            assert set(os.listdir(os.getcwd())) == set([trainer._models[0].model_name, 'other_name.model'])

            # Load
            d2v_loaded = model_storage.load_model(trainer._models[0].model_name)
            assert d2v_loaded.metadata["embedding_model"]["model_type"] == list(trainer.metadata["embedding_trainer"]['models'].values())[0]['embedding_model']['model_type']

            # Change the store directory
            new_path = os.path.join(td, 'other_directory')
            trainer.save_model(FSStore(new_path))
            assert set(os.listdir(new_path)) == set([trainer._models[0].model_name])

    @mock.patch('os.getcwd')
    def test_embedding_trainer_word2vec_local(self, mock_getcwd):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']

        with tempfile.TemporaryDirectory() as td:
            mock_getcwd.return_value = td
            model_storage = ModelStorage(FSStore(td))
            job_postings_generator = JobPostingCollectionSample(num_records=30)
            corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
            w2v = Word2VecModel(size=10, min_count=3, iter=4, window=6, workers=3)

            trainer = EmbeddingTrainer(w2v, model_storage=model_storage)
            trainer.train(corpus_generator)
            trainer.save_model()

            vocab_size = len(w2v.wv.vocab.keys())

            assert w2v.model_name == trainer._models[0].model_name
            assert set(os.listdir(os.getcwd())) == set([trainer._models[0].model_name])

            # Test Online Training
            job_postings_generator = JobPostingCollectionSample(num_records=50)
            corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)

            w2v_loaded =  model_storage.load_model(w2v.model_name)

            new_trainer = EmbeddingTrainer(w2v_loaded, model_storage=model_storage)
            new_trainer.train(corpus_generator)
            new_trainer.save_model()

            new_vocab_size = len(w2v_loaded.wv.vocab.keys())

            assert set(os.listdir(os.getcwd())) == set([trainer._models[0].model_name, new_trainer._models[0].model_name])
            assert new_trainer.metadata['embedding_trainer']['models'] != trainer.metadata['embedding_trainer']['models']
            assert vocab_size <= new_vocab_size

            # Save as different name
            model_storage.save_model(w2v, 'other_name.model')
            assert set(os.listdir(os.getcwd())) == set([trainer._models[0].model_name, new_trainer._models[0].model_name, 'other_name.model'])

            # Change the store directory
            new_path = os.path.join(td, 'other_directory')
            new_trainer.save_model(FSStore(new_path))
            assert set(os.listdir(new_path)) == set([new_trainer._models[0].model_name])

    @mock.patch('os.getcwd')
    def test_embedding_trainer_fasttext_local(self, mock_getcwd):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        with tempfile.TemporaryDirectory() as td:
            mock_getcwd.return_value = td
            model_storage = ModelStorage(FSStore(td))
            job_postings_generator = JobPostingCollectionSample(num_records=30)
            corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
            fasttext = FastTextModel(size=10, min_count=3, iter=4, window=6, workers=3)

            trainer = EmbeddingTrainer(fasttext, model_storage=model_storage)
            trainer.train(corpus_generator)
            trainer.save_model()

            vocab_size = len(fasttext.wv.vocab.keys())

            assert fasttext.model_name == trainer._models[0].model_name
            assert set(os.listdir(os.getcwd())) == set([trainer._models[0].model_name])

            # Test Online Training
            job_postings_generator = JobPostingCollectionSample(num_records=50)
            corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)

            fasttext_loaded = model_storage.load_model(fasttext.model_name)
            new_trainer = EmbeddingTrainer(fasttext_loaded, model_storage=model_storage)
            new_trainer.train(corpus_generator)
            new_trainer.save_model()

            new_vocab_size = len(fasttext_loaded.wv.vocab.keys())

            assert set(os.listdir(os.getcwd())) == set([trainer._models[0].model_name, new_trainer._models[0].model_name])
            assert new_trainer.metadata['embedding_trainer']['models'] != trainer.metadata['embedding_trainer']['models']
            assert vocab_size <= new_vocab_size

    @mock.patch('os.getcwd')
    def test_embedding_trainer_multicore_local(self, mock_getcwd):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        with tempfile.TemporaryDirectory() as td:
            mock_getcwd.return_value = td
            model_storage = ModelStorage(FSStore(td))
            job_postings_generator = JobPostingCollectionSample()
            corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
            trainer = EmbeddingTrainer(
                    FastTextModel(size=10, min_count=3, iter=4, window=6, workers=3),
                    FastTextModel(size=10, min_count=3, iter=4, window=10, workers=3),
                    Word2VecModel(size=10, workers=3, window=6),
                    Word2VecModel(size=10, min_count=10, window=10, workers=3),
                    model_storage=model_storage)
            trainer.train(corpus_generator, n_processes=4)
            trainer.save_model()

            assert set(os.listdir(os.getcwd())) == set([model.model_name for model in trainer._models])

    @mock_s3
    def test_embedding_trainer_multicore_s3(self):
        client = boto3.client('s3')
        client.create_bucket(Bucket='fake-open-skills', ACL='public-read-write')
        s3_path = f"s3://fake-open-skills/model_cache/embedding"
        s3_storage = S3Store(path=s3_path)
        model_storage = ModelStorage(s3_storage)

        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        job_postings_generator = JobPostingCollectionSample()
        corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)
        trainer = EmbeddingTrainer(
                FastTextModel(size=10, min_count=3, iter=4, window=6, workers=3),
                FastTextModel(size=10, min_count=3, iter=4, window=10, workers=3),
                Word2VecModel(size=10, workers=3, window=6),
                Word2VecModel(size=10, min_count=10, window=10, workers=3),
                model_storage=model_storage)
        trainer.train(corpus_generator)
        trainer.save_model()

        s3 = s3fs.S3FileSystem()
        files = [f.split('/')[-1] for f in s3.ls(s3_path)]
        assert set(files) == set([model.model_name for model in trainer._models])

    def test_embedding_trainer_doc2vec_with_other(self):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        job_postings_generator = JobPostingCollectionSample(num_records=30)
        corpus_generator = Doc2VecGensimCorpusCreator(job_postings_generator, document_schema_fields=document_schema_fields)

        trainer = EmbeddingTrainer(Doc2VecModel(), Word2VecModel(), FastTextModel())
        self.assertRaises(TypeError, lambda: trainer.train(corpus_generator))
