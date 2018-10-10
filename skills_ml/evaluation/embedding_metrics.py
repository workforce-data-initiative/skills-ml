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
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        pass

    def eval(self, processing_pipeline, clustering):
        result = {}
        for concept, entities in clustering.items():
            centroid = np.average([processing_pipeline.compose(entity) for entity in entities], axis=0)
            result[concept] = distance.cosine(processing_pipeline.compose(concept), centroid)
        self.eval_result = result
        return result


class IntraClusterCohesion(BaseEmbeddingMetric):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        pass

    def eval(self, processing_pipeline, clustering):
        result = {}
        for concept, entities in clustering.items():
            centroid = np.average([processing_pipeline.compose(entity) for entity in entities], axis=0)
            entities_vec = [processing_pipeline.compose(e) for e in entities]
            result[concept] = np.sum((entities_vec - centroid)**2)
        self.eval_result = result
        return result


class RecallTopN(BaseEmbeddingMetric):
    def __init__(self, topn=20):
        super().__init__()
        self.topn = topn

    def entity_pool(self, clustering):
        pool = []
        for p in clustering.values():
            pool.extend(p)
        return pool

    @property
    def name(self):
        pass

    def eval(self, processing_pipeline, clustering):
        wv = convert_to_keyedvector(self.entity_pool(clustering), processing_pipeline)
        result = {}
        for concept, entities in clustering.items():
            closest = wv.similar_by_vector(processing_pipeline.compose(concept), topn=self.topn)
            tp = set([c[0] for c in closest]) & set([e.identifier for e in entities])
            result[concept] = len(tp) / len(entities)
        return result


class PrecisionTopN(BaseEmbeddingMetric):

    def __init__(self, topn=10):
        self.topn = topn

    def entity_pool(self, clustering):
        pool = []
        for p in clustering.values():
            pool.extend(p)
        return pool

    @property
    def name(self):
        pass

    def eval(self, processing_pipeline, clustering):
        wv = convert_to_keyedvector(self.entity_pool(clustering), processing_pipeline)
        result = {}
        for concept, entities in clustering.items():
            closest = wv.similar_by_vector(processing_pipeline.compose(concept), topn=self.topn)
            tp = set([c[0] for c in closest]) & set([e.identifier for e in entities])
            result[concept] = len(tp) / self.topn
        return result


def convert_to_keyedvector(list_of_occupation, processing_pipeline):
    vector_size = processing_pipeline.functions[-1].keywords['embedding_model'].vector_size
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'w2v.format'), 'w') as f:
            f.write(str(len(list_of_occupation)) + " " + str(vector_size) + "\n")
            for t in list_of_occupation:
                f.write(t.identifier + " ")
                for s in list(map(str, list(processing_pipeline.compose(t.name)))):
                    f.write(s + " ")
                f.write("\n")
        wv = KeyedVectors.load_word2vec_format(os.path.join(td, 'w2v.format'))
    return wv
