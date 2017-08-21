from collections import Counter, defaultdict


class JobAggregator(object):
    def __init__(self, output_count=1, output_total=False):
        self.initialize_counts()
        self.output_count = output_count
        self.output_total = output_total

    def initialize_counts(self):
        self.group_values = defaultdict(Counter)
        self.rollup = defaultdict(Counter)

    def __iadd__(self, other):
        assert type(self) == type(other)
        for key, value in other.group_values.items():
            self.group_values[key] += value
        for key, value in other.rollup.items():
            self.rollup[key] += value

        return self

    def accumulate(self, job_posting, job_key, groups):
        """Incorporates the data from a single job posting

        Args:
            job_posting (dict) a job postings in common schema format
            job_key (str) an element of the job posting (say, a title) that is
                being used for aggregation
            groups (iterable) the aggregatable groups (say, CBSAs)
                that this job posting belongs to
        """
        value = self.value(job_posting)
        for group_key in groups:
            full_key = (group_key, job_key)
            self.group_values[full_key] += value
        self.rollup[job_key] += value

    def value(self, job_posting):
        raise NotImplementedError

    def _outputs(self, row_counter):
        outputs = [
            value
            for value, _ in row_counter.most_common(self.output_count)
        ]
        for _ in range(len(outputs), self.output_count):
            outputs.append(None)
        if self.output_total:
            outputs.append(len(row_counter.keys()))
        return outputs

    def output_header_row(self, prefix):
        output_header_row = [
            '{}_{}'.format(prefix, str(i))
            for i in range(1, self.output_count+1)
        ]
        if self.output_total:
            output_header_row.append('{}_total'.format(prefix))
        return output_header_row

    def group_outputs(self, full_key):
        return self._outputs(self.group_values[full_key])

    def rollup_outputs(self, job_key):
        return self._outputs(self.rollup[job_key])


class CountAggregator(JobAggregator):
    """Counts job postings"""

    def value(self, job_posting):
        return Counter(total=1)

    def output_header_row(self, prefix):
        return ['{}_total'.format(prefix)]

    def _outputs(self, row_counter):
        return [row_counter['total']]


class SkillAggregator(JobAggregator):
    """Aggregates skills found in job postings

    Args:
        skill_extractor (.skill_extractors.FreetextSkillExtractor)
            an object that returns skill counts from unstructured text
        corpus creator (object) an object that returns a text corpus
            from a job posting
    """
    def __init__(self, skill_extractor, corpus_creator, *args, **kwargs):
        super(SkillAggregator, self).__init__(*args, **kwargs)
        self.skill_extractor = skill_extractor
        self.corpus_creator = corpus_creator

    def value(self, job_posting):
        return self.skill_extractor.document_skill_counts(
            self.corpus_creator._transform(job_posting)
        )


class OccupationScopedSkillAggregator(SkillAggregator):
    """Aggregates skills found in job postings using the job's occupation

    Args:
        skill_extractor (.skill_extractors.FreetextSkillExtractor)
            an object that returns skill counts from unstructured text
        corpus creator (object) an object that returns a text corpus
            from a job posting
    """
    def value(self, job_posting):
        return self.skill_extractor.document_skill_counts(
            soc_code=job_posting.get('onet_soc_code', None),
            document=self.corpus_creator._transform(job_posting)
        )


class SocCodeAggregator(JobAggregator):
    """Aggregates SOC codes inferred from job posting text

    Args:
        occupation_classifier (.occupation_classifiers.SocClassifier)
            An object that returns a classified SOC code and similarity score
            from unstructured text
        corpus creator (object) an object that returns unstructured text
            from a job posting
    """
    def __init__(self, occupation_classifier, corpus_creator, *args, **kwargs):
        super(SocCodeAggregator, self).__init__(*args, **kwargs)
        self.occupation_classifier = occupation_classifier
        self.corpus_creator = corpus_creator

    def value(self, job_posting):
        soc_code, similarity_score = self.occupation_classifier.classify(
            self.corpus_creator._transform(job_posting)
        )
        return Counter({soc_code: 1})


class GivenSocCodeAggregator(JobAggregator):
    """Aggregates SOC codes given as an input field

    Caution! We may or may not know where these came from, and the method
    for creating them may differ from record to record
    """
    def value(self, job_posting):
        return Counter({job_posting.get('onet_soc_code', '99-9999.00'): 1})
