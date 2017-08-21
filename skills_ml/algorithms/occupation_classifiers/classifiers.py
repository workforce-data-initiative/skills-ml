import tempfile
import os
import logging
from collections import Counter, defaultdict
import filelock

from gensim.similarities.index import AnnoyIndexer

from skills_ml.algorithms.occupation_classifiers import base
from skills_utils.s3 import download, split_s3_path, list_files


def download_ann_classifier_files(s3_prefix, classifier_id, download_directory, s3_conn):
    lock = filelock.FileLock(os.path.join(download_directory, 'ann_dl.lock'))
    with lock.acquire(timeout=1000):
        s3_path = s3_prefix + classifier_id
        files = list_files(s3_conn, s3_path)
        for f in files:
            filepath = os.path.join(download_directory, f)
            if not os.path.exists(filepath):
                logging.info('calling download from %s to %s', s3_path + f, filepath)
                download(s3_conn, filepath, os.path.join(s3_path, f))
            else:
                logging.info('%s already exists, not downloading', filepath)


class Classifier(object):
    """The Classifiers Object to classify each jobposting description to O*Net SOC code.

    Example:

    from airflow.hooks import S3Hook
    from skills_ml.algorithms.occupation_classifiers.classifiers import Classifier

    s3_conn = S3Hook().get_conn()
    Soc = Classifier(s3_conn=s3_conn, classifier_id='ann_0614')

    predicted_soc = Soc.classify(jobposting, mode='top')
    """
    def __init__(self, classifier_id='ann_0614', classifier=None,
        s3_conn=None, s3_path='open-skills-private/model_cache/', classify_kwargs=None, temporary_directory=None, **kwargs):
        """Initialization of Classifier

        Attributes:
            classifier_id (str): classifier id
            classifier_type (str): classifier type
            s3_path (str): the path of the classifier on S3.
            s3_conn (:obj: `boto.s3.connection.S3Connection`): the boto object to connect to S3.
            files (:obj: `list` of (str)): classifier files need to be downloaded/loaded.
            classifier (:obj): classifier object that will do the actually classification
            classify_kwargs (:dict): arguments to pass through to the .classify method when called on a job posting
        """
        self.classifier_id = classifier_id
        self.classifier_type = classifier_id.split('_')[0]
        self.s3_conn = s3_conn
        self.s3_path = s3_path + classifier_id
        self.files  = list_files(self.s3_conn, self.s3_path)
        self.temporary_directory = temporary_directory or tempfile.TemporaryDirectory()
        self.classifier = self._load_classifier(**kwargs) if classifier == None else classifier
        self.classify_kwargs = classify_kwargs if classify_kwargs else {}

    def _load_classifier(self, **kwargs):
        if self.classifier_type == 'ann':
            for f in self.files:
                filepath = os.path.join(self.temporary_directory, f)
                if not os.path.exists(filepath):
                    logging.warning('calling download from %s to %s', self.s3_path + f, filepath)
                    download(self.s3_conn, filepath, os.path.join(self.s3_path, f))
            ann_index = AnnoyIndexer()
            ann_index.load(os.path.join(self.temporary_directory, self.classifier_id + '.index'))
            return NearestNeighbors(s3_conn=self.s3_conn, indexer=ann_index, **kwargs)

        elif self.classifier_type == 'knn':
            return NearestNeighbors(s3_conn=self.s3_conn, indexed=False, **kwargs)

        else:
            print('Not implemented yet!')
            return None

    def classify(self, jobposting):
        return self.classifier.predict_soc(jobposting, **(self.classify_kwargs))


class NearestNeighbors(base.VectorModel):
    """Nearest neightbors model to classify the jobposting data into soc code.
    If the indexer is passed, then NearestNeighbors will use approximate nearest
    neighbor approach which is much faster than the built-in knn in gensim.

    Attributes:
        indexed (bool): index the data with Annoy or not. Annoy can find approximate nearest neighbors much faster.
        indexer (:obj: `gensim.similarities.index`): Annoy index object should be passed in for faster query.
    """
    def __init__(self, indexed=False, indexer=None, **kwargs):
        super(NearestNeighbors, self).__init__(**kwargs)
        self.indexed = indexed
        self.indexer = self._ann_indexer() if indexed else indexer


    def _ann_indexer(self):
        """This function should be in the training process. It's here for temporary usage.
        Annoy is an open source library to search for points in space that are close to a given query point.
        It also creates large read-only file-based data structures that are mmapped into memory so that many
        processes may share the same data. For our purpose, it is used to find similarity between words or
        documents in a vector space.

        Returns:
            Annoy index object if self.indexed is True. None if we want to use gensim built-in index.
        """

        logging.info('indexing the model %s', self.model_name)
        self.model.init_sims()
        annoy_index = AnnoyIndexer(self.model, 200)
        return annoy_index


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
