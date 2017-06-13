import numpy as np
import logging
from collections import Counter, defaultdict

from gensim.models import Doc2Vec
from skills_ml.algorithms.occupation_classifiers import base


class NearestNeighbors(base.VectorModel):
    """Nearest neightbors model to classify the jobposting data into soc code.
    If the indexer is passed, then NearestNeighbors will use approximate nearest
    neighbor approach which is much faster than the built-in knn in gensim.

    Attributes:
        indexed (bool): index the data with Annoy or not. Annoy can find approximate nearest neighbors much faster.
        indexer (:obj: `gensim.similarities.index`): Annoy index object should be passed in for faster query.
        _training_data (np.ndarray): a document vector array where each row is a document vector.
        _target (np.ndarray): a label array.
    """
    def __init__(self, indexed=False, **kwargs):
        super(NearestNeighbors, self).__init__(**kwargs)
        self.indexed = indexed
        self.indexer = self._ann_indexer()
        self._training_data = self.model.docvecs.doctag_syn0
        self._target = self._create_target_data()


    def _create_target_data(self):
        """To create a label array by mapping each doc vector to the lookup table.

        Returns:
            np.ndarray: label array.
        """
        y = []
        for i in range(len(self._training_data)):
            y.append(self.lookup[self.model.docvecs.index_to_doctag(i)])

        return np.array(y)

    def _ann_indexer(self):
        """Annoy is an open source library to search for points in space that are close to a given query point.
        It also creates large read-only file-based data structures that are mmapped into memory so that many
        processes may share the same data. For our purpose, it is used to find similarity between words or
        documents in a vector space.

        Returns:
            Annoy index object if self.indexed is True. None if we want to use gensim built-in index.
        """
        if self.indexed:
            try:
                from gensim.similarities.index import AnnoyIndexer
            except ImportError:
                raise ValueError("SKIP: Please install the annoy indexer")

            logging.info('indexing the model %s', self.model_name)
            self.model.init_sims()
            annoy_index = AnnoyIndexer(self.model, 200)
            return annoy_index
        else:
            return None


    def predict_soc(self, jobposting, mode='top'):
        """The method to predict the soc code a job posting belongs to.

        Args:
            jobposting (str): a string of cleaned, lower-cased and pre-processed job description context.
            mode (str): a flag of which method to use for classifying.

        Returns:
            tuple(str, float): The predicted soc code and cosine similarity.
        """
        inferred_vector = self.model.infer_vector(jobposting.split())
        if mode == 'top':
            sims = self.model.docvecs.most_similar([inferred_vector], topn=1, indexer=self.indexer)
            resultlist = list(map(lambda l: (self.lookup[l[0]], l[1]), [(x[0], x[1]) for x in sims]))
            predicted_soc = resultlist[0]
            return predicted_soc

        if mode == 'common':
            sims = self.model.docvecs.most_similar([inferred_vector], topn=10, indexer=self.indexer)
            resultlist = list(map(lambda l: (self.lookup[l[0]], l[1]), [(x[0], x[1]) for x in sims]))
            most_common = Counter([r[0] for r in resultlist]).most_common()[0]
            resultdict = defaultdict(list)
            for k, v in resultlist:
                resultdict[k].append(v)

            predicted_soc = (most_common[0], sum(resultdict[most_common[0]])/most_common[1])
            return predicted_soc

