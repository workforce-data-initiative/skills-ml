from skills_ml.algorithms.string_cleaners.nlp import title_phase_one, fields_join, clean_html, clean_str
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
import unittest

class TestStringCleaners(unittest.TestCase):
    def setUp(self):
        self.jp = list(JobPostingCollectionSample())

    def test_title_phase_one(self):
        assert title_phase_one('engineer') == 'engineer'
        assert title_phase_one('engineer/apply now') == 'engineer apply now'
        assert title_phase_one('engineer / apply now') == 'engineer apply now'
        assert title_phase_one("macy's engineer / apply now") == 'macys engineer apply now'

    def test_clean_html(self):
        assert clean_html("<h1>apply now <p>engineer</p></h1>") == 'apply now engineer'

    def test_fields_join(self):
        joined = fields_join(self.jp[0], document_schema_fields=['description', 'experienceRequirements'])
        assert len(joined) == len(' '.join([self.jp[0]['description'], self.jp[0]['experienceRequirements']]))

    def test_deep_wrapper(self):
        assert clean_str([["macy's engineer / apply now", "engineer/apply now"], ["engineer.", "python!"]]) == \
            [["macy s engineer apply now", "engineer apply now"], ["engineer ", "python "]]


