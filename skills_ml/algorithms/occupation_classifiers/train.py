from gensim.models.doc2vec import Doc2Vec
from gensim import __version__ as gensim_version
from gensim import __name__ as gensim_name

from skills_ml.datasets.job_postings import job_postings, job_postings_chain
from skills_ml.algorithms.corpus_creators.basic import Doc2VecGensimCorpusCreator

from skills_utils.s3 import upload

from datetime import datetime
from glob import glob

from itertools import tee

import os
import json
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Reiterable(object):
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        self.iterable, t = tee(self.iterable)
        return t

class RepresentationTrainer(object):
    """A representation learning object using gensim doc2vec model.
    Example:

    from airflow.hooks import S3Hook
    from skills_ml.algorithms.occupation_classifiers.train import RepresentationTrainer

    s3_conn = S3Hook().get_conn()
    trainer = RepresentationTrainer(s3_conn, ['2011Q1', '2011Q2'], 'open-skills-private/test_corpus')
    trainer.train()
    """
    def __init__(self, s3_conn, quarters, jp_s3_path, model_s3_path='open-skills-private/model_cache'):
        """Initialization

        Attributes:
            s3_conn (:obj: `boto.s3.connection.S3Connection`): the boto object to connect to S3.
            quarters (:obj: `list` of (str)): quarters will be trained on
            jp_s3_path (:str): job posting path on S3
            model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model object
            metadata (:dict): model metadata
            training_time (:str): training time
        """
        self.s3_conn = s3_conn
        self.quarters = quarters
        self.jp_s3_path = jp_s3_path
        self.model = None
        self.training_time = datetime.today().isoformat()
        self.model_s3_path = model_s3_path
        self.modelname = 'doc2vec_' + self.training_time
        self.model_path = 'tmp/' + self.modelname + '.model'
        self.lookupname = 'lookup_' + self.training_time
        self.lookup_path = 'tmp/' + self.lookupname + '.json'

    def train(self, size=500, min_count=3, iter=4, window=6, workers=3, **kwargs):
        """Train a doc2vec model, build a lookup table and model metadata. After training, they will be saved to S3.

        Args:
            kwargs: all arguments that gensim.models.doc2vec.Docvec will take.
        """
        job_postings_generator = job_postings_chain(self.s3_conn, self.quarters, self.jp_s3_path)
        corpus = Doc2VecGensimCorpusCreator(job_postings_generator)
        reiterable_corpus = Reiterable(corpus)
        model = Doc2Vec(size=size, min_count=min_count, iter=iter, window=window, workers=workers, **kwargs)
        model.build_vocab(reiterable_corpus)
        model.train(reiterable_corpus, total_examples=model.corpus_count, epochs=model.iter)

        if not os.path.exists('tmp'):
            os.makedirs('tmp')

        self.model = model
        model.save(self.model_path)
        with open(self.lookup_path, 'w') as handle:
            json.dump(corpus.lookup, handle)


        meta_dict = self.metadata
        metaname = 'metadata_' + self.training_time
        meta_path = 'tmp/' + metaname + '.json'
        with open(meta_path, 'w') as handle:
            json.dump(meta_dict, handle, indent=4, separators=(',', ': '))

        for f in glob('tmp/*{}*'.format(self.training_time)):
            upload(self.s3_conn, f, os.path.join(self.model_s3_path, self.modelname))

    @property
    def metadata(self):
        meta_dict = {}
        if self.model:
            meta_dict['metadata'] = {}
            meta_dict['metadata']['hyperparameters'] = {
                                            'vector_size': self.model.vector_size,
                                            'window': self.model.window,
                                            'min_count': self.model.min_count,
                                            'workers': self.model.workers,
                                            'sample': self.model.sample,
                                            'alpha': self.model.alpha,
                                            'seed': self.model.seed,
                                            'iter': self.model.iter,
                                            'hs': self.model.hs,
                                            'negative': self.model.negative,
                                            'dm_mean': self.model.dm_mean if 'dm_mean' in self.model else None,
                                            'cbow_mean': self.model.cbow_mean if 'cbow_mean' in self.model else None,
                                            'dm': self.model.dm,
                                            'dbow_words': self.model.dbow_words,
                                            'dm_concat': self.model.dm_concat,
                                            'dm_tag_count': self.model.dm_tag_count
                                            }
            meta_dict['metadata']['quarters'] = self.quarters
            meta_dict['metadata']['model_name'] = 'doc2vec' + self.training_time
            meta_dict['metadata']['gensim_version']  = gensim_name + gensim_version
            meta_dict['metadata']['training_time'] = self.training_time

        else:
            print("Need to train first")

        return meta_dict
