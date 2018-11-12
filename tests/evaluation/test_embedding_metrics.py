from skills_ml.evaluation.embedding_metrics import metrics_for_embedding, CategorizationMetric, IntraClusterCohesion, RecallTopN, PrecisionTopN
from skills_ml.algorithms.embedding.models import Word2VecModel
from skills_ml.algorithms.preprocessing import ProcessingPipeline
from skills_ml.algorithms import nlp
from skills_ml.ontologies.onet import Onet

import unittest
from functools import partial
from descriptors import cachedproperty
import pytest

# class TestEmbeddingMetrics(unittest.TestCase):
    # @cachedproperty

def vectorization():
    w2v = Word2VecModel(size=10)

    p = ProcessingPipeline(
            nlp.normalize,
            nlp.word_tokenize,
            partial(nlp.vectorize, embedding_model=w2v))
    return p

# @cachedproperty
@pytest.fixture
def major_group_occupation_clustering(onet):
    return onet.major_group_occupation_name_clustering

def test_fix(onet, major_group_occupation_clustering):
    assert onet.major_group_occupation_name_clustering == major_group_occupation_clustering
# def test_categorization_metric():
#     categorization_metric = CategorizationMetric(.major_group_occupation_clustering)
#     result = categorization_metric.eval(.vectorization)

#     assert categorization_metric.name == "major_group_occupations_name_categorization_metric"
#     assert set(result.keys()) == set(.major_group_occupation_clustering.keys())
#     assert 0 <= sum(list(result.values())) / len(result.values()) <= 1

# def test_intra_cluster_cohesion():
#     intra_cluster_cohesion = IntraClusterCohesion(.major_group_occupation_clustering)
#     result = intra_cluster_cohesion.eval(.vectorization)

#     assert intra_cluster_cohesion.name == "major_group_occupations_name_intra_cluster_cohesion"
#     assert set(result.keys()) == set(.major_group_occupation_clustering.keys())
#     assert 0 <= sum(list(result.values())) / len(result.values())

# def test_recall_top_n():
#     recall_top_10 = RecallTopN(.major_group_occupation_clustering, 10)
#     result = recall_top_10.eval(.vectorization)

#     assert recall_top_10.name == "major_group_occupations_name_recall_top10"
#     assert set(result.keys()) == set(.major_group_occupation_clustering.keys())
#     assert 0 <= sum(list(result.values())) / len(result.values()) <= 1

# def test_precision_top_n():
#     precision_top_10 = PrecisionTopN(.major_group_occupation_clustering, 10)
#     result = precision_top_10.eval(.vectorization)

#     assert precision_top_10.name == "major_group_occupations_name_precision_top10"
#     assert set(result.keys()) == set(.major_group_occupation_clustering.keys())
#     assert 0 <= sum(list(result.values())) / len(result.values()) <= 1

# def test_metrics_for_embedding():
#     metric_objects = [
#             CategorizationMetric(.major_group_occupation_clustering),
#             IntraClusterCohesion(.major_group_occupation_clustering),
#             RecallTopN(.major_group_occupation_clustering, 10),
#             PrecisionTopN(.major_group_occupation_clustering, 10)
#     ]
#     full_result = metrics_for_embedding(.vectorization, metric_objects)
#     assert set(full_result.keys()) == set([metric.name for metric in metric_objects])
