import boto
import boto3
import json
from moto import mock_s3, mock_s3_deprecated
import unittest
from unittest.mock import patch
import importlib
import inspect
import functools

from tests import utils

from skills_ml.job_postings.computed_properties.computers import (
    PostingIdPresent,
    TitleCleanPhaseOne,
    TitleCleanPhaseTwo,
    CBSAandStateFromGeocode,
    ClassifyTop,
    ExactMatchSkillCounts
)


class ComputedPropertyTestCase(unittest.TestCase):
    datestring = '2016-01-01'

    def test_aggregator_compatibility(self):
        """Test whether or not the computer's compatible_aggregate_function_paths
            are in fact compatible with pandas.DataFrame.agg
        """
        computed_property = getattr(self, 'computed_property', None)
        if not computed_property:
            if self.__class__.__name__ == 'ComputedPropertyTestCase':
                return
            else:
                raise ValueError('All subclasses of ComputedPropertyTestCase ' +
                                 'should create self.computed_property in self.setUp')

        df = self.computed_property.df_for_date(self.datestring)

        def pandas_ready_functions(paths):
            """Generate aggregate functions from the configured aggregate function paths

            Suitable for testing whether or not the functions work with pandas

            Yields: callables
            """
            if paths:
                for path in paths.keys():
                    module_name, func_name = path.rsplit(".", 1)
                    module = importlib.import_module(module_name)
                    func = getattr(module, func_name)
                    # skills_ml.algorithms.aggregators.pandas functions are wrapped
                    if hasattr(func, 'function'):
                        base_func = func.function
                    else:
                        base_func = func

                    # assuming here that the first arg will be a configurable number,
                    # e.g. top 'n'
                    if len(inspect.getargspec(base_func).args) == 2:
                        yield functools.partial(func, 2)
                    else:
                        yield func

        for column in self.computed_property.property_columns:
            for func in pandas_ready_functions(column.compatible_aggregate_function_paths):
                df.agg(func)


@mock_s3
class PostingIdPresentTest(ComputedPropertyTestCase):
    def setUp(self):
        self.client = boto3.resource('s3')
        self.client.create_bucket(Bucket='test-bucket')
        self.computed_property = PostingIdPresent(path='test-bucket/computed_properties')
        self.job_postings = [utils.job_posting_factory(datePosted=self.datestring)]
        self.computed_property.compute_on_collection(self.job_postings)

    def test_compute_func(self):
        cache = self.computed_property.cache_for_date(self.datestring)
        job_posting_id = self.job_postings[0]['id']
        assert cache[str(job_posting_id)] == 1

    def test_sum(self):
        pass


@mock_s3
class TitleCleanPhaseOneTest(ComputedPropertyTestCase):
    def setUp(self):
        self.client = boto3.resource('s3')
        self.client.create_bucket(Bucket='test-bucket')
        self.computed_property = TitleCleanPhaseOne(path='test-bucket/computed_properties')
        self.job_postings = [utils.job_posting_factory(datePosted=self.datestring, title='Software Engineer - Tulsa')]
        self.computed_property.compute_on_collection(self.job_postings)

    def test_compute_func(self):
        cache = self.computed_property.cache_for_date(self.datestring)
        job_posting_id = self.job_postings[0]['id']
        assert cache[str(job_posting_id)] == 'software engineer tulsa'


@mock_s3
class TitleCleanPhaseTwoTest(ComputedPropertyTestCase):
    def setUp(self):
        self.client = boto3.resource('s3')
        self.client.create_bucket(Bucket='test-bucket')
        self.computed_property = TitleCleanPhaseTwo(path='test-bucket/computed_properties')
        self.job_postings = [utils.job_posting_factory(datePosted=self.datestring, title='Software Engineer Tulsa')]
        with patch('skills_ml.algorithms.jobtitle_cleaner.clean.negative_positive_dict', return_value={'places': ['tulsa'], 'states': [], 'onetjobs': ['software engineer']}):
            self.computed_property.compute_on_collection(self.job_postings)

    def test_compute_func(self):
        cache = self.computed_property.cache_for_date(self.datestring)
        job_posting_id = self.job_postings[0]['id']
        assert cache[str(job_posting_id)] == 'software engineer'



@mock_s3
class CBSAAndStateFromGeocodeTest(ComputedPropertyTestCase):
    def setUp(self):
        client = boto3.resource('s3')
        bucket = client.create_bucket(Bucket='test-bucket')
        sample_cbsa_cache = {
            'AMENIA, North Dakota': ['22020', 'Fargo, ND-MN Metro Area']
        }
        bucket.put_object(Key='cbsas.json', Body=json.dumps(sample_cbsa_cache))
        self.computed_property = CBSAandStateFromGeocode(
            cache_s3_path='test-bucket/cbsas',
            path='test-bucket/computed_properties',
        )
        self.job_postings = [utils.job_posting_factory(datePosted=self.datestring, jobLocation={"@type": "Place", "address": {"addressLocality": "AMENIA", "addressRegion": "ND", "@type": "PostalAddress"}} )]
        self.computed_property.compute_on_collection(self.job_postings)

    @patch('skills_ml.datasets.cbsa_shapefile.download_shapefile')
    def test_compute_func(self, download_mock):
        cache = self.computed_property.cache_for_date(self.datestring)
        job_posting_id = self.job_postings[0]['id']
        assert cache[job_posting_id] == ['22020', 'Fargo, ND-MN Metro Area', 'ND']


@mock_s3_deprecated
@mock_s3
class SocClassifyTest(ComputedPropertyTestCase):
    def setUp(self):
        s3_conn = boto.connect_s3()
        client = boto3.resource('s3')
        bucket = client.create_bucket(Bucket='test-bucket')
        description = 'This is my description'
        class MockClassifier(object):
            def predict_soc(self, document, mode):
                assert document.strip() == description.lower()
                assert mode == 'top'
                return '11-1234.00'

        self.computed_property = ClassifyTop(
            s3_conn=s3_conn,
            classifier_obj=MockClassifier(),
            path='test-bucket/computed_properties',
        )
        self.job_postings = [utils.job_posting_factory(datePosted=self.datestring, description=description, skills='', qualifications='', experienceRequirements='')]
        self.computed_property.compute_on_collection(self.job_postings)

    def test_compute_func(self):
        cache = self.computed_property.cache_for_date(self.datestring)
        job_posting_id = self.job_postings[0]['id']
        assert cache[job_posting_id] == '11-1234.00'


@mock_s3
@mock_s3_deprecated
class SkillExtractTest(ComputedPropertyTestCase):
    def setUp(self):
        s3_conn = boto.connect_s3()
        client = boto3.resource('s3')
        bucket = client.create_bucket(Bucket='test-bucket')
        skills_path = 's3://test-bucket/skills_master_table.tsv'
        utils.create_skills_file(skills_path)
        self.computed_property = ExactMatchSkillCounts(
            skill_lookup_path=skills_path,
            path='test-bucket/computed_properties',
        )
        self.job_postings = [utils.job_posting_factory(
            datePosted=self.datestring,
            description='reading comprehension'
        )]
        self.computed_property.compute_on_collection(self.job_postings)

    def test_compute_func(self):
        cache = self.computed_property.cache_for_date(self.datestring)
        job_posting_id = self.job_postings[0]['id']
        assert cache[job_posting_id] == {'skill_counts_exact_match': ['reading comprehension']}
