from skills_ml.algorithms.occupation_classifiers.train import OccupationClassifierTrainer, create_training_set
from skills_ml.algorithms.occupation_classifiers import SOCMajorGroup
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
    def setUp(self):
        self.embedding_model = None
        self.jobpostings = None
        self.train_embedding()

    def has_soc_filter(self, document):
        if document['onet_soc_code'][:2] != '99' and document['onet_soc_code'] != '':
            return True
        else:
             return False

    def train_embedding(self):
        jobpostings = list(JobPostingCollectionSample())
        corpus_generator = Word2VecGensimCorpusCreator(jobpostings, raw=True)
        w2v = Word2VecModel(size=10, min_count=0, alpha=0.025, min_alpha=0.025)
        trainer = EmbeddingTrainer(corpus_generator, w2v)
        trainer.train(True)

        self.embedding_model = w2v
        self.jobpostings = jobpostings

    def test_create_training_set(self):
        jp_f = list(JobPostingFilterer(self.jobpostings, [self.has_soc_filter]))
        matrix = create_training_set(jp_f, SOCMajorGroup(), self.embedding_model)
        assert matrix.target_variable.name == "major_group"
        assert matrix.X.shape[0] ==  len(jp_f)
        assert matrix.y.shape[0] == len(jp_f)
        assert matrix.embedding_model == self.embedding_model
        assert matrix.target_variable.encoder.inverse_transform([0]) == '11'

    def test_training(self):
        jp_f = JobPostingFilterer(self.jobpostings, [self.has_soc_filter])
        matrix = create_training_set(jp_f, SOCMajorGroup(), self.embedding_model)
        assert matrix.target_variable.name == "major_group"

        occ_trainer = OccupationClassifierTrainer(matrix, k_folds=2, grid_config=grid, scoring=['accuracy'])
        occ_trainer.train()
        assert list(occ_trainer.cls_cv_result['accuracy'].keys()) == ['ExtraTreesClassifier']
        assert occ_trainer.matrix.embedding_model.model_name == self.embedding_model.model_name

    def test_filtering(self):
        major_group_27_filter = lambda job: job['onet_soc_code'][:2] != '27'
        major_group_49_filter = lambda job: job['onet_soc_code'][:2] != '49'

        soc_target = SOCMajorGroup()
        matrix = create_training_set(self.jobpostings, soc_target, self.embedding_model,)
        assert '27' in matrix.target_variable.encoder.inverse_transform(matrix.y)

        soc_target = SOCMajorGroup(major_group_27_filter)
        matrix = create_training_set(self.jobpostings, soc_target, self.embedding_model)
        assert '27' not in matrix.target_variable.encoder.inverse_transform(matrix.y)

        soc_target = SOCMajorGroup([major_group_27_filter, major_group_49_filter])
        matrix = create_training_set(self.jobpostings, soc_target, self.embedding_model)
        assert '27' not in matrix.target_variable.encoder.inverse_transform(matrix.y)
        assert '49' not in matrix.target_variable.encoder.inverse_transform(matrix.y)

