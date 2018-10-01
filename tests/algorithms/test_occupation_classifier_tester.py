from skills_ml.algorithms.occupation_classifiers.classifiers import CombinedClassifier
from skills_ml.algorithms.occupation_classifiers.train import OccupationClassifierTrainer
from skills_ml.algorithms.occupation_classifiers.test import OccupationClassifierTester
from skills_ml.algorithms.occupation_classifiers import DesignMatrix, FullSOC
from skills_ml.algorithms.embedding.models import Word2VecModel
from skills_ml.algorithms.embedding.train import EmbeddingTrainer
from skills_ml.algorithms.string_cleaners import nlp
from skills_ml.algorithms.preprocessing import IterablePipeline

from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.job_postings.corpora import Word2VecGensimCorpusCreator

from sklearn.ensemble import RandomForestClassifier

from descriptors import cachedproperty

from functools import partial
from itertools import islice
import unittest

class TestOccupationClassifierTester(unittest.TestCase):
    @cachedproperty
    def fullsoc(self):
        return FullSOC()

    @property
    def pipe_x(self):
        document_schema_fields = ['description', 'experienceRequirements', 'qualifications', 'skills']
        pipe_x = IterablePipeline(
                self.fullsoc.filter,
                partial(nlp.fields_join, document_schema_fields=document_schema_fields),
                nlp.clean_str,
                nlp.word_tokenize,
                partial(nlp.vectorize, embedding_model=Word2VecModel(size=10))
        )
        return pipe_x

    @property
    def pipe_y(self):
        pipe_y = IterablePipeline(
                self.fullsoc.filter,
                self.fullsoc.transformer
        )
        return pipe_y

    @property
    def grid_config(self):
        return {
                'sklearn.ensemble.RandomForestClassifier': {
                    'n_estimators': [5, 10],
                    'criterion': ['entropy'],
                    'max_depth': [1],
                    'max_features': ['log2'],
                    'min_samples_split': [5]
                    }
                }

    def test_tester(self):
        document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']
        corpus_generator = Word2VecGensimCorpusCreator(JobPostingCollectionSample(num_records=30), document_schema_fields=document_schema_fields)
        w2v = Word2VecModel(size=10, min_count=3, iter=4, window=6, workers=3)
        trainer = EmbeddingTrainer(corpus_generator, w2v)
        trainer.train()

        jp = JobPostingCollectionSample()
        train_gen = islice(jp, 30)
        test_gen = islice(jp, 30, None)
        train_matrix = DesignMatrix(train_gen, self.fullsoc, self.pipe_x, self.pipe_y)
        train_matrix.build()
        occ_trainer = OccupationClassifierTrainer(train_matrix, 2, grid_config=self.grid_config)
        occ_trainer.train(save=False)
        print(occ_trainer.best_estimators[0].target_variable)
        cc = CombinedClassifier(w2v, occ_trainer.best_estimators[0])

        steps = self.pipe_x.generators[:-1]

        test_gen = (t for t in test_gen if t['onet_soc_code'] is not '')

        tester = OccupationClassifierTester(test_data_generator=test_gen, preprocessing=steps, classifier=cc)
        result = list(tester)

        assert len(tester) == len(result) == 18
