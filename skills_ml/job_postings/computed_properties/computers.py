"""Various computers of job posting properties. Each class is generally a generic algorithm (such as skill extraction or occupation classification) paired with enough configuration to run on its own"""
import json
import tempfile

from . import JobPostingComputedProperty, ComputedPropertyColumn

from skills_ml.algorithms.string_cleaners import NLPTransforms
from skills_ml.algorithms.jobtitle_cleaner.clean import JobTitleStringClean
from skills_ml.algorithms.occupation_classifiers.classifiers import \
    Classifier
from skills_ml.job_postings.corpora.basic import SimpleCorpusCreator
from skills_ml.job_postings.geography_queriers.cbsa_from_geocode import JobCBSAFromGeocodeQuerier
from skills_ml.algorithms.skill_extractors import\
    ExactMatchSkillExtractor, SocScopedExactMatchSkillExtractor
from skills_ml.algorithms.geocoders.cbsa import CachedCBSAFinder


class TitleCleanPhaseOne(JobPostingComputedProperty):
    """Perform one phase of job title cleaning: lowercase/remove punctuation"""
    property_name = 'title_clean_phase_one'
    property_columns = [
        ComputedPropertyColumn(
            name='title_clean_phase_one',
            description='Job title, cleaned by lowercasing and removing punctuation'
        )
    ]

    def _compute_func_on_one(self):
        title_func = NLPTransforms().title_phase_one
        return lambda job_posting: title_func(job_posting['title'])


class TitleCleanPhaseTwo(JobPostingComputedProperty):
    """Perform two phases of job title cleaning:

    1. lowercase/remove punctuation
    2. Remove geography information
    """
    property_name = 'title_clean_phase_two'
    property_columns = [
        ComputedPropertyColumn(
            name='title_clean_phase_two',
            description='Job title, cleaned by lowercasing, removing punctuation, and removing geography information',
        )
    ]

    def _compute_func_on_one(self):
        return lambda job_posting: JobTitleStringClean().clean_title(NLPTransforms().title_phase_one(job_posting['title']))


class CBSAandStateFromGeocode(JobPostingComputedProperty):
    """Produce a CBSA by geocoding the job's location and matching with a CBSA shapefile

    Args:
        cache_s3_path (string) An s3 path to store geocode cache results
    """
    def __init__(self, cache_storage, cache_fname, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_storage = cache_storage
        self.cache_fname = cache_fname

    property_name = 'cbsa_and_state_from_geocode'
    property_columns = [
        ComputedPropertyColumn(
            name='cbsa_fips',
            description='FIPS code of Core-Based Statistical Area, found by geocoding job location'
        ),
        ComputedPropertyColumn(
            name='cbsa_name',
            description='Name of Core-Based Statistical Area, found by geocoding job location'
        ),
        ComputedPropertyColumn(
            name='state_code',
            description='State code of job posting'
        )
    ]

    def _compute_func_on_one(self):
        geo_querier = JobCBSAFromGeocodeQuerier(
            cbsa_results=CachedCBSAFinder(
                cache_storage=self.cache_storage,
                cache_fname=self.cache_fname
            ).all_cached_cbsa_results
        )
        return lambda job_posting: geo_querier.query(job_posting)


class SOCClassifyProperty(JobPostingComputedProperty):
    """Classify the SOC code from a trained classifier

    Args:
        s3_conn (boto.s3.connection)
        classifier_obj (object, optional) An object to use as a classifier.
            If not sent one will be downloaded from s3
    """
    def __init__(self, s3_conn, classifier_obj=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s3_conn = s3_conn
        self.temp_dir = tempfile.TemporaryDirectory()
        self.classifier_obj = classifier_obj

    def _compute_func_on_one(self):
        common_classifier = Classifier(
            s3_conn=self.s3_conn,
            classifier_id='ann_0614',
            classify_kwargs=self.classify_kwargs,
            classifier=self.classifier_obj,
            temporary_directory=self.temp_dir
        )
        corpus_creator = SimpleCorpusCreator()

        def func(job_posting):
            return common_classifier.classify(corpus_creator._transform(job_posting))

        return func


class ClassifyCommon(SOCClassifyProperty):
    """Classify SOC code using common match method"""
    classify_kwargs = {'mode': 'common'}
    property_name = 'soc_common'
    property_columns = [
        ComputedPropertyColumn(
            name='soc_common',
            description='SOC code inferred by common match method',
            compatible_aggregate_function_paths={
                'skills_ml.job_postings.aggregate.pandas.n_most_common': 'Most common'
            }
        )
    ]


class ClassifyTop(SOCClassifyProperty):
    """Classify SOC code using top match method"""
    classify_kwargs = {'mode': 'top'}
    property_name = 'soc_top'
    property_columns = [
        ComputedPropertyColumn(
            name='soc_top',
            description='SOC code inferred by top match method',
            compatible_aggregate_function_paths={
                'skills_ml.job_postings.aggregate.pandas.n_most_common': 'Most common'
            }
        )
    ]


class GivenSOC(JobPostingComputedProperty):
    """Assign the SOC code given by the partner"""
    property_name = 'soc_given'
    property_columns = [
        ComputedPropertyColumn(
            name='soc_given',
            description='SOC code given by partner',
            compatible_aggregate_function_paths={
                'skills_ml.job_postings.aggregate.pandas.n_most_common': 'Most common'
            }
        )
    ]

    def _compute_func_on_one(self):
        def func(job_posting):
            return job_posting.get('onet_soc_code', '99-9999.00')
        return func


class SocScopedExactMatchSkillCounts(JobPostingComputedProperty):
    """Find skills by exact matching, and filter out ones that aren't associated
    with the job's given SOC code

    Args:
        skill_lookup_path (string) A path to a skills lookup file, containing both
            skill names and their association with SOC codes. See
            SocScopedExactMatchSkillExtractor for more details
    """
    def __init__(self, skill_lookup_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_lookup_path = skill_lookup_path

    property_name = 'skill_counts_soc_scoped'
    property_columns = [
        ComputedPropertyColumn(
            name='skill_counts_soc_scoped',
            description='ONET skills found by exact matching that are present in the skillset indicated by the given SOC code',
            compatible_aggregate_function_paths={
                'skills_ml.job_postings.aggregate.pandas.listy_n_most_common': 'Most common'
            }
        )
    ]

    def _compute_func_on_one(self):
        corpus_creator = SimpleCorpusCreator()
        skill_extractor = SocScopedExactMatchSkillExtractor(
            skill_lookup_path=self.skill_lookup_path
        )

        def func(job_posting):
            count_dict = skill_extractor.document_skill_counts(
              soc_code=job_posting.get('onet_soc_code', '99-9999.00'),
              document=corpus_creator._transform(job_posting)
            )
            count_lists = [[k] * v for k, v in count_dict.items()]
            flattened = [count for countlist in count_lists for count in countlist]
            return {self.property_name: flattened}
        return func


class ExactMatchSkillCounts(JobPostingComputedProperty):
    """Find skills by exact matching

    Args:
        skill_lookup_path (string) A path to a skills lookup file, containing skill names.
            See ExactMatchSkillExtractor for more details
    """
    def __init__(self, skill_lookup_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_lookup_path = skill_lookup_path

    property_name = 'skill_counts_exact_match'
    property_columns = [
        ComputedPropertyColumn(
            name='skill_counts_exact_match',
            description='ONET skills found by exact matching',
            compatible_aggregate_function_paths={
                'skills_ml.job_postings.aggregate.pandas.listy_n_most_common': 'Most common'
            }
        )
    ]

    def _compute_func_on_one(self):
        corpus_creator = SimpleCorpusCreator()
        skill_extractor = ExactMatchSkillExtractor(skill_lookup_path=self.skill_lookup_path)

        def func(job_posting):
            count_dict = skill_extractor.document_skill_counts(
              document=corpus_creator._transform(job_posting)
            )
            count_lists = [[k] * v for k, v in count_dict.items()]
            flattened = [count for countlist in count_lists for count in countlist]
            return {self.property_name: flattened}
        return func


class PostingIdPresent(JobPostingComputedProperty):
    """Records job posting ids. Used for counting job postings"""
    property_name = 'posting_id_present'
    property_columns = [
        ComputedPropertyColumn(
            name='posting_id_present',
            description='Job postings',
            compatible_aggregate_function_paths={'numpy.sum': 'Count of'}
        )
    ]

    def _compute_func_on_one(self):
        return lambda job_posting: 1
