import unittest
import numpy as np
from skills_ml.evaluation.occ_cls_evaluator import ClassificationEvaluator, OnetOccupationClassificationEvaluator
from skills_ml.algorithms.occupation_classifiers import FullSOC

class FakeResultGenerator(object):
    def __init__(self):
        self.y_true = [0, 1, 200, 0, 1, 200, 0, 1, 200, 0, 1, 200]
        self.y_pred = [0, 1, 200, 1, 200, 0, 1, 1, 1, 0, 1, 0]
        self.target_variable = FullSOC()

    def __iter__(self):
        for r in zip(self.y_pred, self.y_true):
            yield r

class TestOccupationClassifierEvaluator(unittest.TestCase):
    def test_classification_evaluator(self):
        evaluator = ClassificationEvaluator(result_generator=FakeResultGenerator())

        p = evaluator.precision[np.nonzero(evaluator.precision)]
        r = evaluator.recall[np.nonzero(evaluator.recall)]

        assert evaluator.accuracy == 0.5
        self.assertSequenceEqual(list(p), [0.5, 0.5 , 0.5])
        self.assertSequenceEqual(list(r), [0.5 , 0.75, 0.25])
        self.assertSequenceEqual(list(evaluator.f1[np.nonzero(evaluator.f1)]), list(2 * r * p / (r + p)))
        assert evaluator.micro_precision == 0.5
        assert evaluator.micro_recall == 0.5
        assert evaluator.macro_precision == 0.5
        assert evaluator.macro_recall == 0.5
        assert evaluator.micro_f1 == 0.5
        self.assertAlmostEqual(evaluator.macro_f1, 0.477, places=2)

    def test_onet_occupation_classification_evaulator(self):
        onet_evaluator = OnetOccupationClassificationEvaluator(result_generator=FakeResultGenerator())

        r = onet_evaluator.recall_per_major_group
        p =  onet_evaluator.precision_per_major_group

        self.assertAlmostEqual(onet_evaluator.accuracy_major_group, 0.6666666, places=2)
        self.assertSequenceEqual(list(onet_evaluator.recall_per_major_group), [0.875, 0.25 ])
        self.assertSequenceEqual(list(onet_evaluator.precision_per_major_group), [0.7, 0.5])
        self.assertSequenceEqual(list(onet_evaluator.f1_per_major_group), list(2 * r * p / (r + p)))
