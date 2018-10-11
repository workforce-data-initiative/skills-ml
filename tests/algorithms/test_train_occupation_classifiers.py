from skills_ml.algorithms.occupation_classifiers.train import OccupationClassifierTrainer
from skills_ml.algorithms.occupation_classifiers import SOCMajorGroup, FullSOC, DesignMatrix
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.job_postings.filtering import JobPostingFilterer
from skills_ml.job_postings.corpora import Word2VecGensimCorpusCreator
from skills_ml.algorithms.embedding.models import Word2VecModel
from skills_ml.algorithms.embedding.train import EmbeddingTrainer
from skills_ml.algorithms.string_cleaners import nlp
from skills_ml.algorithms.preprocessing import IterablePipeline
from skills_ml.storage import FSStore, ModelStorage

import os
import time
import tempfile
import unittest
import mock
from descriptors import cachedproperty
from functools import partial
from sklearn.externals import joblib


grid = {
        'sklearn.ensemble.ExtraTreesClassifier': {
            'n_estimators': [100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [1, 5],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [5]
            }
        }


class TestClassifierTrainer(unittest.TestCase):
    def setUp(self):
        self.jobpostings = list(JobPostingCollectionSample())

    def basic_filter(self, doc):
        if self.major_group.filter_func(doc):
            return doc
        else:
            return None

    @property
    def pipe_x(self):
        document_schema_fields = ['description', 'experienceRequirements', 'qualifications', 'skills']
        pipe_x = IterablePipeline(
                self.basic_filter,
                partial(nlp.fields_join, document_schema_fields=document_schema_fields),
                nlp.clean_str,
                nlp.word_tokenize,
                partial(nlp.vectorize, embedding_model=self.embedding_model)
                )
        return pipe_x

    @property
    def pipe_y(self):
        pipe_y = IterablePipeline(
                self.basic_filter,
                self.major_group.transformer
                )
        return pipe_y

    def has_soc_filter(self, document):
        if document['onet_soc_code'][:2] != '99' and document['onet_soc_code'] != '':
            return True
        else:
             return False

    @cachedproperty
    def major_group(self):
        return SOCMajorGroup()

    @cachedproperty
    def full_soc(self):
        return FullSOC()

    @cachedproperty
    def embedding_model(self):
        w2v = Word2VecModel(size=10)
        return w2v

    @cachedproperty
    def matrix(self):
        jp_f = JobPostingFilterer(self.jobpostings, [self.has_soc_filter])
        matrix = DesignMatrix(jp_f, self.major_group, self.pipe_x, self.pipe_y)
        matrix.build()
        return matrix

    @property
    def filtered_jobpostings(self, filters=None):
        if not filters:
            filters = [self.has_soc_filter]
        return list(JobPostingFilterer(self.jobpostings, filters))

    def test_full_soc(self):
        job_posting = {'onet_soc_code': '11-1031.00', 'id': 'some_id_number'}
        assert self.full_soc.transformer(job_posting)[0] == [3]

    def test_design_matrix(self):
        matrix = self.matrix
        assert matrix.target_variable.name == "major_group"
        assert matrix.X.shape[0] ==  len(self.filtered_jobpostings)
        assert matrix.y.shape[0] == len(self.filtered_jobpostings)
        assert matrix.target_variable.encoder.inverse_transform([0]) == '11'

    def test_training_not_save(self):
        matrix = self.matrix
        assert matrix.target_variable.name == "major_group"
        occ_trainer = OccupationClassifierTrainer(matrix, k_folds=2, grid_config=grid, scoring=['accuracy'])
        occ_trainer.train(save=False)
        assert list(occ_trainer.cls_cv_result['accuracy'].keys()) == ['ExtraTreesClassifier']

    @mock.patch('os.getcwd')
    def test_training_save(self, mock_getcwd):
        with tempfile.TemporaryDirectory() as td:
            mock_getcwd.return_value = td
            matrix = self.matrix
            assert matrix.target_variable.name == "major_group"
            occ_trainer = OccupationClassifierTrainer(
                    matrix,
                    k_folds=2,
                    storage=FSStore(td),
                    grid_config=grid,
                    scoring=['accuracy'])
            occ_trainer.train(save=True)
            assert set(os.listdir(os.getcwd())) ==  set([occ_trainer.train_time])

    def test_one_filter(self):
        # First make sure 27 is in the data
        matrix = self.matrix
        assert '27' in matrix.target_variable.encoder.inverse_transform(matrix.y)

        # Create a filter to prune 27
        major_group_27_filter = lambda job: job['onet_soc_code'][:2] != '27'
        soc_target = SOCMajorGroup(major_group_27_filter)
        def new_filter(doc):
            if soc_target.filter_func(doc):
                return doc
            else:
                return None

        document_schema_fields = ['description', 'experienceRequirements', 'qualifications', 'skills']
        pipe_x = IterablePipeline(
                 new_filter,
                 partial(nlp.fields_join, document_schema_fields=document_schema_fields),
                 nlp.clean_str,
                 nlp.word_tokenize,
                 partial(nlp.vectorize, embedding_model=self.embedding_model)
                 )

        pipe_y = IterablePipeline(
                 new_filter,
                 soc_target.transformer
                 )

        matrix = DesignMatrix(self.jobpostings, soc_target, pipe_x, pipe_y)
        matrix.build()
        assert '27' not in matrix.target_variable.encoder.inverse_transform(matrix.y)


    def test_two_filters(self):
        major_group_27_filter = lambda job: job['onet_soc_code'][:2] != '27'
        major_group_49_filter = lambda job: job['onet_soc_code'][:2] != '49'
        soc_target = SOCMajorGroup([major_group_27_filter, major_group_49_filter])

        def new_filter(doc):
             if soc_target.filter_func(doc):
                 return doc
             else:
                 return None

        document_schema_fields = ['description', 'experienceRequirements', 'qualifications', 'skills']
        pipe_x = IterablePipeline(
                  new_filter,
                  partial(nlp.fields_join, document_schema_fields=document_schema_fields),
                  nlp.clean_str,
                  nlp.word_tokenize,
                  partial(nlp.vectorize, embedding_model=self.embedding_model)
                  )

        pipe_y = IterablePipeline(
                  new_filter,
                  soc_target.transformer
                  )

        matrix = DesignMatrix(self.jobpostings, soc_target, pipe_x, pipe_y)
        matrix.build()
        assert '27' not in matrix.target_variable.encoder.inverse_transform(matrix.y)
        assert '49' not in matrix.target_variable.encoder.inverse_transform(matrix.y)

