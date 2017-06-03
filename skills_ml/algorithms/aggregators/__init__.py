from collections import Counter, defaultdict


class JobAggregator(object):
    def __init__(self):
        self.initialize_counts()

    def initialize_counts(self):
        self.group_values = defaultdict(Counter)
        self.rollup = defaultdict(Counter)

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


class CountAggregator(JobAggregator):
    """Counts job postings"""

    def initialize_counts(self):
        self.group_values = Counter()
        self.rollup = Counter()

    def value(self, job_posting):
        return 1


class SkillAggregator(JobAggregator):
    """Aggregates skills found in job postings

    Args:
        skill_extractor (.skill_extractors.FreetextSkillExtractor)
            an object that returns skill counts from unstructured text
        corpus creator (object) an object that returns a text corpus
            from a job posting
    """
    def __init__(self, skill_extractor, corpus_creator):
        super(SkillAggregator, self).__init__()
        self.skill_extractor = skill_extractor
        self.corpus_creator = corpus_creator

    def value(self, job_posting):
        return self.skill_extractor.document_skill_counts(
            self.corpus_creator._transform(job_posting)
        )
