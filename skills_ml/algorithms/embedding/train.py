import gensim.models.doc2vec
assert gensim.models.doc2vec.FAST_VERSION > -1

from skills_ml.job_postings.common_schema import batches_generator
from skills_ml.algorithms.embedding.models import Word2VecModel

from datetime import datetime
from itertools import tee

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Reiterable(object):
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        self.iterable, t = tee(self.iterable)
        return t


class EmbeddingTrainer(object):
    """An embedding learning class.
    Example:

    ```python
    from skills_ml.algorithms.occupation_classifiers.train import EmbeddingTrainer
    from skills_ml.job_postings.common_schema import JobPostingCollectionSample
    from skills_ml.job_postings.corpora.basic import Doc2VecGensimCorpusCreator, Word2VecGensimCorpusCreator
    from skills_ml.storage import FSStore

    model = Word2VecModel(size=size, min_count=min_count, iter=iter, window=window, workers=workers, **kwargs)

    s3_conn = S3Hook().get_conn()
    job_postings_generator = JobPostingGenerator(s3_conn, quarters, s3_path, source="all")
    corpus_generator = Word2VecGensimCorpusCreator(job_postings_generator)
    w2v = Word2VecModel(storage=FSStore(path='/tmp'), size=10, min_count=3, iter=4, window=6, workers=3)
    trainer = EmbeddingTrainer(corpus_generator, w2v)
    trainer.train()
    trainer.save()
    ```
    """
    def __init__(
        self, corpus_generator, model, batch_size=2000):
        """Initialization

        Attributes:
            corpus_generator (:generator): the iterable corpus
            storage (:obj: `skills_ml.Store`): skills_ml Store object
            metadata (:dict): model metadata
            training_time (:str): training time
            batch_size (:int): batch size
            vocab_size_cumu (:list): record the number of vocab every batch for word2vec
            _model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model object
            lookup (:dict): dictionary for storing the training documents and keys for doc2vec
        """
        self.corpus_generator = corpus_generator
        self.training_time = datetime.today().isoformat()
        self.update = False
        self.batch_size = batch_size
        self.vocab_size_cumu = []
        self.lookup_dict = None
        self._model = model
        self.model_type = model.model_type

    def train(self, lookup=False, *args, **kwargs):
        """Train an embedding model, build a lookup table and model metadata. After training, they will be saved to S3.

        Args:
            kwargs: all arguments that gensim.models will take.
        """
        if self.model_type in ['word2vec', 'fasttext']:
            if self._model.wv.vocab:
                logging.info("Model has been trained")
                self.update = True

            batch_iter = 1
            batch_gen = batches_generator(self.corpus_generator, self.batch_size)
            for batch in batch_gen:
                batch = Reiterable(batch)
                logging.info("Training batch #{} ".format(batch_iter))
                if not self.update:
                    self._model.build_vocab(batch, update=False)
                    self.update = True
                else:
                    self._model.build_vocab(batch, update=True)

                self._model.train(batch, total_examples=self._model.corpus_count, epochs=self._model.iter, *args, **kwargs)
                self.vocab_size_cumu.append(len(self._model.wv.vocab))
                batch_iter += 1
                logging.info('\n')

        elif self.model_type == 'doc2vec':
            corpus_gen = self.corpus_generator
            reiter_corpus_gen = Reiterable(corpus_gen)
            self._model.build_vocab(reiter_corpus_gen)
            self._model.train(reiter_corpus_gen, total_examples=self._model.corpus_count, epochs=self._model.iter, *args, **kwargs)
            if lookup:
                self.lookup_dict = corpus_gen.lookup
                self._model.lookup_dict = self.lookup_dict

        self._model.model_name = self.model_name

    def save_model(self, storage=None):
        if storage is None:
            self._model.save(self.model_name)
        else:
            self._model.storage = storage
            self._model.save(self.model_name)
        logging.info(f"{self.model_name} has been saved.")

    @property
    def model_name(self):
        return self.model_type + '_' + self.training_time + '.model'

    @property
    def metadata(self):
        meta_dict = {'embedding_trainer': {}}
        if self._model:
            meta_dict['embedding_trainer']['model_name'] = self.model_name
            meta_dict['embedding_trainer']['training_time'] = self.training_time
            meta_dict['embedding_trainer']['vocab_size_cumu'] = self.vocab_size_cumu
        else:
            print("Haven't trained the model yet!")

        try:
            meta_dict.update(self.corpus_generator.metadata)
            meta_dict.update(self._model.metadata)
        except AttributeError:
            logging.info(self.corpus_generator.__class__.__name__ + " has no metadata!")

        return meta_dict
