from skills_ml.ontologies.clustering import Clustering

from scipy.spatial import distance
from gensim.models import KeyedVectors
import numpy as np
import tempfile
import os


class BaseEmbeddingMetric(object):
    def __init__(self):
        self.eval_result = None

    def eval(self):
        return NotImplementedError


class CategorizationMetric(BaseEmbeddingMetric):
    def __init__(self, clustering: Clustering):
        self.clustering = clustering
        super().__init__()

    @property
    def name(self):
        return f"{self.clustering.name}_categorization_metric"

    def eval(self, vectorizing_pipeline):
        result = {}
        for concept, entities in self.clustering.items():
            centroid = np.average([vectorizing_pipeline.compose(self.clustering.value_transform_fn(entity)) for entity in entities], axis=0)
            result[concept] = distance.cosine(vectorizing_pipeline.compose(concept), centroid)
        self.eval_result = result
        return result


class IntraClusterCohesion(BaseEmbeddingMetric):
    def __init__(self, clustering: Clustering):
        self.clustering = clustering
        super().__init__()

    @property
    def name(self):
        return f"{self.clustering.name}_intra_cluster_cohesion"

    def eval(self, vectorizing_pipeline):
        result = {}
        for concept, entities in self.clustering.items():
            entities_vec = [vectorizing_pipeline.compose(self.clustering.value_transform_fn(entity)) for entity in entities]
            centroid = np.average(entities_vec, axis=0)
            result[concept] = np.sum((entities_vec - centroid)**2)
        self.eval_result = result
        return result


class RecallTopN(BaseEmbeddingMetric):
    def __init__(self, clustering: Clustering, topn=20):
        self.clustering = clustering
        self.topn = topn
        super().__init__()

    @property
    def entity_pool(self):
        pool = []
        for p in self.clustering.values():
            pool.extend(p)
        return pool

    @property
    def name(self):
        return f"{self.clustering.name}_recall_top{self.topn}"

    def eval(self, vectorizing_pipeline):
        wv = convert_to_keyedvector(
                self.entity_pool,
                vectorizing_pipeline,
                value_transform_fn=self.clustering.value_transform_fn
        )
        result = {}
        for concept, entities in self.clustering.items():
            closest = wv.similar_by_vector(vectorizing_pipeline.compose(concept), topn=self.topn)
            tp = set([c[0] for c in closest]) & set([e.identifier for e in entities])
            result[concept] = len(tp) / len(entities)
        return result


class PrecisionTopN(BaseEmbeddingMetric):
    def __init__(self, clustering: Clustering, topn=10):
        self.clustering = clustering
        self.topn = topn
        super().__init__()

    @property
    def entity_pool(self):
        pool = []
        for p in self.clustering.values():
            pool.extend(p)
        return pool

    @property
    def name(self):
        return f"{self.clustering.name}_precision_top{self.topn}"

    def eval(self, vectorizing_pipeline):
        wv = convert_to_keyedvector(
                self.entity_pool,
                vectorizing_pipeline,
                value_transform_fn=self.clustering.value_transform_fn
        )
        result = {}
        for concept, entities in self.clustering.items():
            closest = wv.similar_by_vector(vectorizing_pipeline.compose(concept), topn=self.topn)
            tp = set([c[0] for c in closest]) & set([e.identifier for e in entities])
            result[concept] = len(tp) / self.topn
        return result


def convert_to_keyedvector(
        list_of_objects,
        vectorizing_pipeline,
        value_transform_fn=lambda entity: entity,
        identifier_fn=lambda key: getattr(key, "identifier"),
        ):
    vector_size = vectorizing_pipeline.functions[-1].keywords['embedding_model'].vector_size
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'w2v.format'), 'w') as f:
            f.write(str(len(list_of_objects)) + " " + str(vector_size) + "\n")
            for obj in list_of_objects:
                f.write(identifier_fn(obj) + " ")
                for s in list(map(str, list(vectorizing_pipeline.compose(value_transform_fn(obj))))):
                    f.write(s + " ")
                f.write("\n")
        wv = KeyedVectors.load_word2vec_format(os.path.join(td, 'w2v.format'))
    return wv
