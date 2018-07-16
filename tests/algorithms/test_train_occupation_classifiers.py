from skills_ml.algorithms.occupation_classifiers.train import OccupationClassifierTrainer, create_training_set
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.job_postings.filtering import JobPostingFilterer
from skills_ml.job_postings.corpora.basic import Word2VecGensimCorpusCreator
from skills_ml.algorithms.embedding.models import Word2VecModel
from skills_ml.algorithms.embedding.train import EmbeddingTrainer
import unittest


grid = {
        'sklearn.ensemble.ExtraTreesClassifier': {
            'n_estimators': [100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [1, 5, 10],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5]
            }
        }

class TestClassifierTrainer(unittest.TestCase):

    def has_soc_filter(self, document):
        if document['onet_soc_code'] != None and document['onet_soc_code'] != '':
            return True
        else:
             return False

    def test_training(self):
        jobpostings = list(JobPostingCollectionSample())
        corpus_generator = Word2VecGensimCorpusCreator(jobpostings, raw=True)
        w2v = Word2VecModel(size=10, min_count=0, alpha=0.025, min_alpha=0.025)
        trainer = EmbeddingTrainer(corpus_generator, w2v)
        trainer.train(True)

        jp_f = JobPostingFilterer(jobpostings, [self.has_soc_filter])
        X, y = create_training_set(jp_f, w2v)

        occ_trainer = OccupationClassifierTrainer(k_folds=2, target_variable='soc', grid_config=grid, scores=['accuracy'])
        occ_trainer.fit(X, y)
        assert list(occ_trainer.cls_cv_result['ExtraTreesClassifier'].keys()) == ['accuracy']




