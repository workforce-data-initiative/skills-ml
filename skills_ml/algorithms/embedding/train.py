import gensim.models.doc2vec
assert gensim.models.doc2vec.FAST_VERSION > -1

from skills_ml.job_postings.common_schema import batches_generator, BatchGenerator
from skills_ml.algorithms.embedding.models import Word2VecModel
from skills_ml.storage import ModelStorage
from skills_ml.utils import filename_friendly_hash

import multiprocess as mp
from datetime import datetime, timedelta
from itertools import tee
from functools import partial
from time import time
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
    trainer = EmbeddingTrainer(w2v)
    trainer.train(corpus_generator)
    trainer.save_model()
    ```
    """
    def __init__(
        self, *models, model_storage=None, batch_size=2000):
        """Initialization

        Attributes:
            corpus_generator (:generator): the iterable corpus
            storage (:obj: `skills_ml.Store`): skills_ml Store object
            metadata (:dict): model metadata
            training_time (:str): training time
            batch_size (:int): batch size
            vocab_size_cumu (:list): record the number of vocab every batch for word2vec
            _model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model object
            lookup_dict (:dict): dictionary for storing the training documents and keys for doc2vec
        """
        self.model_storage = model_storage
        self.training_time = datetime.today().isoformat()
        self.batch_size = batch_size
        self.lookup_dict = None
        self._models = models
        self.model_type = set([model.model_type for model in models])
        self.corpus_metadata = None

    def _train_one_batch(self, model, batch, *args, **kwargs):
        if len(model.wv.vocab) == 0:
            model.build_vocab(batch, update=False)
        else:
            model.build_vocab(batch, update=True)

        model.train(batch, total_examples=model.corpus_count, epochs=model.iter, *args, **kwargs)
        return model

    def train(self, corpus_generator, lookup=False, n_processes=1, *args, **kwargs):
        """Train an embedding model, build a lookup table and model metadata. After training, they will be saved to S3.

        Args:
            kwargs: all arguments that gensim.models will take.
        """
        tic = time()
        try:
            self.corpus_metadata = corpus_generator.metadata
        except AttributeError:
            self.corpus_metadata = None

        if any([model.wv.vocab for model in self._models]):
            logging.info("Model has been trained")
            return 0

        for model in self._models:
            model.model_name = "_".join([
                model.model_type,
                self._model_hash(model, self.training_time, self.corpus_metadata)]
            ) + '.model'

        if self.model_type <= set(['word2vec', 'fasttext']):
            batch_gen = BatchGenerator(corpus_generator, self.batch_size)
            if n_processes == 1:
                for i, batch in enumerate(batch_gen):
                    logging.info("Training batch #{} ".format(i))
                    self._models = [self._train_one_batch(model, batch) for model in self._models]
            else:
                with mp.Pool(processes=n_processes, maxtasksperchild=1) as pool:
                    for i, batch in enumerate(batch_gen):
                        logging.info("Training batch #{} ".format(i))
                        partial_train = partial(self._train_one_batch, batch=batch, *args, **kwargs)
                        self._models = pool.map(partial_train, self._models)

        elif set(self.model_type) == set(['doc2vec']):
            logging.info(f"Training {self.model_type}")
            reiter_corpus_gen = Reiterable(corpus_generator)
            self._models = [self._train_one_batch(model, reiter_corpus_gen) for model in self._models]
            # partial_train = partial(self._train_one_batch, batch=reiter_corpus_gen)
            # with mp.Pool(processes=n_processes, maxtasksperchild=1) as pool:
                # self._models = pool.map(partial_train, self._models)
            if lookup:
                self.lookup_dict = corpus_generator.lookup
                self._models[0].lookup_dict = self.lookup_dict
        else:
            raise TypeError

        logging.info(f"{', '.join([m.model_name for m in self._models])} are trained in {str(timedelta(seconds=time()-tic))}")

    def _model_hash(self, model, training_time, corpus_metadata):
        unique = {
            "model_metadata": model.metadata,
            "training_time": training_time,
            "corpus_metadata": corpus_metadata
        }
        return filename_friendly_hash(unique)

    def save_model(self, storage=None):
        if storage is None:
            if self.model_storage is None:
                raise AttributeError(f"'self.model_storage' should not be None if you want to save the model")
            ms = self.model_storage
            for model in self._models:
                ms.save_model(model, model.model_name)
                logging.info(f"{model.model_name} has been stored to {ms.storage.path}.")
        else:
            ms = ModelStorage(storage)
            for model in self._models:
                ms.save_model(model, model.model_name)
            logging.info(f"{model.model_name} has been stored to {ms.storage.path}.")

    @property
    def metadata(self):
        meta_dict = {'embedding_trainer': {}}
        if self._models:
            meta_dict['embedding_trainer']['models'] = {model.model_name: model.metadata for model in self._models}
            meta_dict['embedding_trainer']['training_time'] = self.training_time
        else:
            print("Haven't trained the model yet!")

        try:
            meta_dict.update(self.corpus_metadata)
        except AttributeError:
            logging.info(self.corpus_generator.__class__.__name__ + " has no metadata!")

        return meta_dict


