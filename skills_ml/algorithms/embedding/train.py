from gensim.models.doc2vec import Doc2Vec, Word2Vec
from gensim import __version__ as gensim_version
from gensim import __name__ as gensim_name
import gensim.models.doc2vec
assert gensim.models.doc2vec.FAST_VERSION > -1

from skills_ml.datasets.job_postings import job_postings_chain, batches_generator
from skills_ml.algorithms.corpus_creators.basic import Doc2VecGensimCorpusCreator, Word2VecGensimCorpusCreator
from skills_ml.algorithms.embedding.base import Word2VecModel

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

    s3_conn = S3Hook().get_conn()
    trainer = EmbeddingTrainer(s3_conn, ['2011Q1', '2011Q2'], 'open-skills-private/test_corpus')
    trainer.train()
    ```
    """
    def __init__(
        self, s3_conn, quarters, jp_s3_path, source='all',
        model_s3_path=S3_PATH_EMBEDDING_MODEL, batch_size=2000,
        model_type='word2vec'):
        """Initialization

        Attributes:
            s3_conn (:obj: `boto.s3.connection.S3Connection`): the boto object to connect to S3.
            quarters (:obj: `list` of (str)): quarters will be trained on
            source (:str): job posting source, should be "all", "nlx" or "cb".
            jp_s3_path (:str): job posting path on S3
            _model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model object
            metadata (:dict): model metadata
            training_time (:str): training time
            batch_size (:int): batch size
            model_type (:str): 'word2vec' or 'doc2vec'
        """
        if model_type not in ['word2vec', 'doc2vec']:
            raise ValueError('"{}"" model_type is not supported!'.format(model_type))
        elif model_type == 'doc2vec':
            logging.warning("Current gensim doc2vec doesn't support online batch learning. Use generic learning instead.")
            logging.warning("Training on too large corpus might cause serious memory leaks!")

        if source not in ['nlx', 'cb', 'all']:
            raise ValueError('"{}" is an invalid source!'.format(source))

        if not isinstance(quarters, list):
            raise TypeError('quarters should be a list of string, e.g. ["2011Q1", "2011Q2"]')

        self.s3_conn = s3_conn
        self.quarters = quarters
        self.jp_s3_path = jp_s3_path
        self.training_time = datetime.today().isoformat()
        self.model_type = model_type
        self.model_s3_path = model_s3_path
        self.modelname = self.model_type + '_gensim_' + self.training_time
        self.source = source
        self.update = False
        self.batch_size = batch_size
        self.vocab_size_cumu = []
        self._model = None
        self._lookup = None

    def load(self, model_name, s3_path=S3_PATH_EMBEDDING_MODEL):
        if 'word2vec' in model_name.split('_'):
            word2vec = Word2VecModel(model_name=model_name, s3_conn=self.s3_conn, s3_path=s3_path)
            self._model = word2vec.model
            self.model_type = 'word2vec'
        elif 'doc2vec' in model_name.split('_'):
            raise NotImplementedError("Couldn't load doc2vec model. Current gensim doc2vec model doesn't support online learning")

    def _upload(self):
        with tempfile.TemporaryDirectory() as td:
            self.save_model(td)
            for f in glob(os.path.join(td, '*{}*'.format(self.training_time))):
                upload(self.s3_conn, f, os.path.join(self.model_s3_path, self.modelname))

    def save_model(self, path):
        """Save model locally

        Args:
            path (:str): path to save model files and lookup
        """
        model_name = self.modelname + '.model'
        self._model.save(os.path.join(path, model_name))
        meta_dict = self.metadata
        metaname = 'metadata_' + self.modelname + '.json'
        with open(os.path.join(path, metaname), 'w') as handle:
            json.dump(meta_dict, handle, indent=4, separators=(',', ': '))

        if self.model_type == 'doc2vec':
            lookup_name = 'lookup_' + self.modelname + '.json'
            with open(os.path.join(path, lookup_name), 'w') as handle:
                json.dump(self._lookup, handle)

    def train(self, size=500, min_count=3, iter=4, window=6, workers=3, **kwargs):
        """Train an embedding model, build a lookup table and model metadata. After training, they will be saved to S3.

        Args:
            kwargs: all arguments that gensim.models.doc2vec.Docvec will take.
        """
        job_postings_generator = job_postings_chain(self.s3_conn, self.quarters, self.jp_s3_path, source=self.source)

        if self.model_type == 'word2vec':
            if not self._model:
                model = Word2Vec(size=size, min_count=min_count, iter=iter, window=window, workers=workers, **kwargs)
            else:
                logging.info("Model existed")
                model = self._model
                self.update = True

            batch_iter = 1
            batch_gen = batches_generator(Word2VecGensimCorpusCreator(job_postings_generator), self.batch_size)
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
            corpus_gen = Doc2VecGensimCorpusCreator(job_postings_generator)
            reiter_corpus_gen = Reiterable(corpus_gen)
            model.build_vocab(reiter_corpus_gen)
            model.train(reiter_corpus_gen, total_examples=model.corpus_count, epochs=model.iter)
            self._lookup = corpus_gen.lookup

        self._model = model
        self._upload()


    @property
    def metadata(self):
        meta_dict = {}
        if self._model:
            meta_dict['metadata'] = {}
            meta_dict['model_name'] = self.modelname
            meta_dict['metadata']['hyperparameters'] = {
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
            meta_dict['metadata']['quarters'] = self.quarters
            meta_dict['metadata']['gensim_version']  = gensim_name + gensim_version
            meta_dict['metadata']['training_time'] = self.training_time
            meta_dict['metadata']['vocab_size_cumu'] = self.vocab_size_cumu

        else:
            print("Need to train first")

        return meta_dict
