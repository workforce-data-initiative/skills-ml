from gensim.models.doc2vec import Doc2Vec, Word2Vec
from gensim import __version__ as gensim_version
from gensim import __name__ as gensim_name
import gensim.models.doc2vec
assert gensim.models.doc2vec.FAST_VERSION > -1

from skills_ml.job_postings.common_schema import batches_generator
from skills_ml.algorithms.embedding.base import Word2VecModel

from skills_ml.storage import FSStore

from skills_utils.s3 import upload

from datetime import datetime
from glob import glob

from itertools import tee

import os
import json
import tempfile
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

S3_PATH_EMBEDDING_MODEL = 'open-skills-private/model_cache/embedding/'

class Reiterable(object):
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        self.iterable, t = tee(self.iterable)
        return t


class EmbeddingTrainer(object):
    """An embedding learning object using gensim word2vec/doc2vec model.
    Example:
    ```
    from airflow.hooks import S3Hook
    from skills_ml.algorithms.occupation_classifiers.train import EmbeddingTrainer
    from skills_ml.job_postings.common_schema import JobPostingGenerator
    from skills_ml.job_postings.corpora.basic import Doc2VecGensimCorpusCreator, Word2VecGensimCorpusCreator

    s3_conn = S3Hook().get_conn()
    job_postings_generator = JobPostingGenerator(s3_conn, quarters, s3_path, source="all")
    corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator)
    trainer = EmbeddingTrainer(corpus_generator, s3_conn, 'open-skills-private/test_corpus')
    trainer.train()
    ```
    """
    def __init__(
        self, corpus_generator, batch_size=2000, storage=FSStore()):
        """Initialization

        Attributes:
            corpus_generator (:generator): the iterable corpus
            storage (:obj: `skills_ml.Store`): skills_ml Store object
            _model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model object
            metadata (:dict): model metadata
            training_time (:str): training time
            batch_size (:int): batch size
            model_type (:str): 'word2vec' or 'doc2vec'
        """
        self.corpus_generator = corpus_generator
        self.training_time = datetime.today().isoformat()
        self.update = False
        self.batch_size = batch_size
        self.vocab_size_cumu = []
        self._model = None
        self._lookup = None
        self.storage = storage

    def load(self, model_name):
        if 'word2vec' in model_name.split('_'):
            word2vec = Word2VecModel(self.storage)
            word2vec.load_model(model_name)
            self._model = word2vec._model
        elif 'doc2vec' in model_name.split('_'):
            raise NotImplementedError("Couldn't load doc2vec model. Current gensim doc2vec model doesn't support online learning")

    def save_model(self):
        """Save model locally

        Args:
            path (:str): path to save model files and lookup
        """
        model_name = self.modelname + '.model'

        self.storage.write(self._model, model_name)

        meta_dict = self.metadata
        metaname = 'metadata_' + self.modelname + '.json'
        self.storage.write(meta_dict, metaname)

        if self.model_type == 'doc2vec':
            lookup_name = 'lookup_' + self.modelname + '.json'
            self.storage.write(self._lookup, lookup_name)

    def train(self, size=500, min_count=3, iter=4, window=6, workers=3, **kwargs):
        """Train an embedding model, build a lookup table and model metadata. After training, they will be saved to S3.

        Args:
            kwargs: all arguments that gensim.models.doc2vec.Docvec will take.
        """
        if self.model_type == 'word2vec':
            if not self._model:
                model = Word2Vec(size=size, min_count=min_count, iter=iter, window=window, workers=workers, **kwargs)
            else:
                logging.info("Model existed")
                model = self._model
                self.update = True

            batch_iter = 1
            batch_gen = batches_generator(self.corpus_generator, self.batch_size)
            for batch in batch_gen:
                batch = Reiterable(batch)
                logging.info("Training batch #{} ".format(batch_iter))
                if not self.update:
                    model.build_vocab(batch, update=False)
                    self.update = True
                else:
                    model.build_vocab(batch, update=True)

                model.train(batch, total_examples=model.corpus_count, epochs=model.iter)
                self.vocab_size_cumu.append(len(model.wv.vocab))
                batch_iter += 1
                logging.info('\n')

        elif self.model_type == 'doc2vec':
            model = Doc2Vec(size=size, min_count=min_count, iter=iter, window=window, workers=workers, **kwargs)
            corpus_gen = self.corpus_generator
            reiter_corpus_gen = Reiterable(corpus_gen)
            model.build_vocab(reiter_corpus_gen)
            model.train(reiter_corpus_gen, total_examples=model.corpus_count, epochs=model.iter)
            self._lookup = corpus_gen.lookup

        self._model = model

    @property
    def model_type(self):
        if self.corpus_generator.__class__.__name__ == 'Doc2VecGensimCorpusCreator':
            return 'doc2vec'
        elif self.corpus_generator.__class__.__name__ == 'Word2VecGensimCorpusCreator':
            return 'word2vec'

    @property
    def modelname(self):
        return self.model_type + '_gensim_' + self.training_time

    @property
    def metadata(self):
        meta_dict = {'embedding_trainer': {}}
        if self._model:
            meta_dict['embedding_trainer']['model_name'] = self.modelname
            meta_dict['embedding_trainer']['hyperparameters'] = {
                                            'vector_size': self._model.vector_size,
                                            'window': self._model.window,
                                            'min_count': self._model.min_count,
                                            'workers': self._model.workers,
                                            'sample': self._model.sample,
                                            'alpha': self._model.alpha,
                                            'seed': self._model.seed,
                                            'iter': self._model.iter,
                                            'hs': self._model.hs,
                                            'negative': self._model.negative,
                                            'dm_mean': self._model.dm_mean if 'dm_mean' in self._model else None,
                                            'cbow_mean': self._model.cbow_mean if 'cbow_mean' in self._model else None,
                                            'dm': self._model.dm if hasattr(self._model, 'dm') else None,
                                            'dbow_words': self._model.dbow_words if hasattr(self._model, 'dbow_words') else None,
                                            'dm_concat': self._model.dm_concat if hasattr(self._model, 'dm_concat') else None,
                                            'dm_tag_count': self._model.dm_tag_count if hasattr(self._model, 'dm_tag_count') else None
                                            }
            meta_dict['embedding_trainer']['gensim_version']  = gensim_name + gensim_version
            meta_dict['embedding_trainer']['training_time'] = self.training_time
            meta_dict['embedding_trainer']['vocab_size_cumu'] = self.vocab_size_cumu

        else:
            print("Haven't trained the model yet!")

        meta_dict.update(self.corpus_generator.metadata)
        return meta_dict
