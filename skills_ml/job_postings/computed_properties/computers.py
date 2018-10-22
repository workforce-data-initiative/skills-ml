"""Various computers of job posting properties. Each class is generally a generic algorithm (such as skill extraction or occupation classification) paired with enough configuration to run on its own"""
from . import JobPostingComputedProperty, ComputedPropertyColumn

from skills_ml.algorithms.nlp import title_phase_one
from skills_ml.algorithms.jobtitle_cleaner.clean import JobTitleStringClean
from skills_ml.algorithms.occupation_classifiers.classifiers import \
    SocClassifier
from skills_ml.job_postings.corpora import SimpleCorpusCreator
import logging
import statistics


NUMERIC_AGGREGATION_FUNCTION_PATHS = {
    'numpy.mean': 'Arithemetic mean',
    'numpy.median': 'Median (middle value)',
    'numpy.std': 'Sample standard deviation',
    'numpy.var': 'Sample variance',
}


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
        title_func = title_phase_one
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
        return lambda job_posting: JobTitleStringClean().clean_title(title_phase_one(job_posting['title']))


class Geography(JobPostingComputedProperty):
    """Produce a geography by querying a given JobGeographyQuerier

    Args:
        geo_querier
    """
    def __init__(self, geo_querier, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geo_querier = geo_querier

    @property
    def property_name(self):
        return self.geo_querier.name

    @property
    def property_columns(self):
        return [
            ComputedPropertyColumn(name=name, description=description)
            for name, description
            in self.geo_querier.output_columns
        ]

    def _compute_func_on_one(self):
        return lambda job_posting: self.geo_querier.query(job_posting)


class SOCClassifyProperty(JobPostingComputedProperty):
    """Classify the SOC code from a trained classifier

    Args:
        classifier_obj (object, optional) An object to use as a classifier.
            If not sent one will be downloaded from s3
    """
    def __init__(self, classifier_obj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = SocClassifier(classifier_obj)

    def _compute_func_on_one(self):

        common_classifier = self.classifier
        corpus_creator = SimpleCorpusCreator()

        def func(job_posting):
            return common_classifier.predict_soc(corpus_creator._transform(job_posting))

        return func

    @property
    def property_name(self):
        return self.classifier.name

    @property
    def property_description(self):
        return self.classifier.description

    @property
    def property_columns(self):
        property_columns = [
            ComputedPropertyColumn(
                name=self.property_name,
                description=self.property_description,
                compatible_aggregate_function_paths={
                    'skills_ml.job_postings.aggregate.pandas.n_most_common': 'Most common'
                }
            )
        ]
        return property_columns


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



class PayMixin(object):
    def salary_from_job_posting(self, job_posting):
        minSalary = None
        maxSalary = None
        try:
            minSalary = float(job_posting['baseSalary']['minValue'])
        except ValueError:
            logging.warning('Could not cast minValue string %s to float', job_posting['baseSalary']['minValue'])

        try:
            maxSalary = float(job_posting['baseSalary']['maxValue'])
        except ValueError:
            logging.warning('Could not cast maxValue string %s to float', job_posting['baseSalary']['maxValue'])

        if not maxSalary and not minSalary:
            logging.warning('Neither minSalary nor maxSalary could be converted to float, no extraction possible')
            return None

        if not maxSalary:
            return minSalary

        if not minSalary:
            return maxSalary

        return statistics.mean([minSalary, maxSalary])


class HourlyPay(JobPostingComputedProperty, PayMixin):
    """The pay given in the baseSalary field if salaryFrequency is hourly"""
    property_name = 'hourly_pay'
    property_columns = [
        ComputedPropertyColumn(
            name='pay_hourly',
            description='Pay given in baseSalary field if salaryFrequency is hourly',
            compatible_aggregate_function_paths=NUMERIC_AGGREGATION_FUNCTION_PATHS,
        )
    ]

    def _compute_func_on_one(self):
        def func(job_posting):
            if job_posting.get('baseSalary', {}).get('salaryFrequency', None) != 'hourly':
                return None
            return self.salary_from_job_posting(job_posting)
        return func


class YearlyPay(JobPostingComputedProperty, PayMixin):
    """The pay given in the baseSalary field if salaryFrequency is yearly"""
    property_name = 'yearly_pay'
    property_columns = [
        ComputedPropertyColumn(
            name='pay_yearly',
            description='Pay given in baseSalary field if salaryFrequency is yearly',
            compatible_aggregate_function_paths=NUMERIC_AGGREGATION_FUNCTION_PATHS,
        )
    ]

    def _compute_func_on_one(self):
        def func(job_posting):
            if job_posting.get('baseSalary', {}).get('salaryFrequency', None) != 'yearly':
                return None
            return self.salary_from_job_posting(job_posting)
        return func


class SkillCounts(JobPostingComputedProperty):
    """Adding top skill counts from a skill extractor

    Args: (skills_ml.algorithms.skill_extractors.base.SkillExtractorBase) A skill extractor object
    """
    def __init__(self, skill_extractor, *args, **kwargs):
        self.skill_extractor = skill_extractor
        super().__init__(*args, **kwargs)

    @property
    def property_name(self):
        return f'skill_counts_{self.skill_extractor.name}'

    @property
    def property_columns(self):
        return [ComputedPropertyColumn(
            name=self.property_name,
            description=self.skill_extractor.description,
            compatible_aggregate_function_paths={
                'skills_ml.job_postings.aggregate.pandas.listy_n_most_common': 'Most common'
            }
        )]

    def _compute_func_on_one(self):
        def func(job_posting):
            count_dict = self.skill_extractor.document_skill_counts(job_posting)
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
