from skills_ml.ontologies.clustering import Clustering
from skills_ml.algorithms.preprocessing import IterablePipeline
from itertools import chain

from collections import defaultdict
from scipy.spatial import distance
from gensim.models import KeyedVectors
from typing import Callable, Dict
import numpy as np
import tempfile
import os


def metrics_for_embedding(vectorization_pipeline, metric_objects):
    result = defaultdict(dict)
    for metric in metric_objects:
        result[metric.name] =  metric.eval(vectorization_pipeline)
    return result


class BaseEmbeddingMetric(object):
    def __init__(self):
        self.eval_result = None

    def eval(self):
        return NotImplementedError


class CategorizationMetric(BaseEmbeddingMetric):
    """cosine similarity between the clustering concept and the mean vector
    of all entities within that concept cluster.
    """
    def __init__(self, clustering: Clustering):
        self.clustering = clustering
        super().__init__()

    @property
    def name(self):
        return f"{self.clustering.name}_categorization_metric"

    def eval(self, vectorization: Callable) -> Dict:
        result = {}
        for concept, entities in self.clustering.items():
            centroid = np.average([vectorization(entity[1]) for entity in entities], axis=0)
            result[concept] = distance.cosine(vectorization(concept), centroid).astype(float)
        self.eval_result = result
        return result


class IntraClusterCohesion(BaseEmbeddingMetric):
    """sum of squared error of the centroid of the concept cluster and
    each entities within the concept cluster.

    """
    def __init__(self, clustering: Clustering):
        self.clustering = clustering
        super().__init__()

    @property
    def name(self):
        return f"{self.clustering.name}_intra_cluster_cohesion"

    def eval(self, vectorization: Callable) -> Dict:
        result = {}
        for concept, entities in self.clustering.items():
            entities_vec = [vectorization(entity[1]) for entity in entities]
            centroid = np.average(entities_vec, axis=0)
            result[concept] = np.sum((entities_vec - centroid)**2).astype(float)
        self.eval_result = result
        return result


class RecallTopN(BaseEmbeddingMetric):
    """For a given concept cluster and a given number n, find top
    n similar entities from the whole entity pool based on cosin
    similarity, and then calculate the top n recall: number of the true
    positive from top n closest entities divided by the total number
    of the concept cluster.
    """
    def __init__(self, clustering: Clustering, topn=20):
        self.clustering = clustering
        self.topn = topn
        super().__init__()

    @property
    def entity_generator(self):
        return chain(*self.clustering.values())

    @property
    def name(self):
        return f"{self.clustering.name}_recall_top{self.topn}"

    def eval(self, vectorization: Callable) -> Dict:
        convert2tuples = IterablePipeline(
                lambda x: (x[0], vectorization(x[1]))
        )
        wv = convert_to_keyedvector(list(convert2tuples(self.entity_generator)))

        result = {}
        for concept, entities in self.clustering.raw_items():
            closest = wv.similar_by_vector(vectorization(concept), topn=self.topn)
            tp = set([c[0] for c in closest]) & set([e.identifier for e in entities])
            result[concept] = len(tp) / len(entities)
        return result


class PrecisionTopN(BaseEmbeddingMetric):
    """For a given concept cluster and a given number n, find top
    n similar entities from the whole entity pool based on cosin
    similarity, and then calculate the top n precision: number of the true
    positive from top n closest entities divided by n.
    """
    def __init__(self, clustering: Clustering, topn=10):
        self.clustering = clustering
        self.topn = topn
        super().__init__()

    @property
    def entity_generator(self):
        return chain(*self.clustering.values())

    @property
    def name(self):
        return f"{self.clustering.name}_precision_top{self.topn}"

    def eval(self, vectorization: Callable) -> Dict:
        convert2tuples = IterablePipeline(
                lambda x: (x[0], vectorization(x[1]))
        )
        wv = convert_to_keyedvector(list(convert2tuples(self.entity_generator)))
        result = {}
        for concept, entities in self.clustering.raw_items():
            closest = wv.similar_by_vector(vectorization(concept), topn=self.topn)
            tp = set([c[0] for c in closest]) & set([e.identifier for e in entities])
            result[concept] = len(tp) / self.topn
        return result


def convert_to_keyedvector(id_vector_tuples):
    vector_size = len(id_vector_tuples[0][1])
    with tempfile.NamedTemporaryFile(mode='w') as f:
        f.write(str(len(id_vector_tuples)) + " " + str(vector_size) + "\n")
        for identifier, vectors in id_vector_tuples:
            f.write(identifier + " ")
            for vector in list(map(str, vectors)):
                f.write(vector + " ")
            f.write("\n")
        f.seek(0)
        return KeyedVectors.load_word2vec_format(f.name)
