from skills_ml.algorithms.preprocessing import IterablePipeline, fields_joiner, \
     html_cleaner, sentence_tokenizer, str_cleaner, word_tokenizer
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.algorithms.string_cleaners.nlp import NLPTransforms

from functools import partial, update_wrapper
import unittest

class TestNLPPipeline(unittest.TestCase):
    def setUp(self):
        self.jp = list(JobPostingCollectionSample())
        self.jp_with_html = [
                "<h1>Overview:</h1><p>St. Michael's Inc. is seeking dedicated federal government " \
                "auditors from junior through the manager level to join our growing team for NGA.</p>"]
        self.nlp = NLPTransforms()

    def test_fields_joiner(self):
        joined = list(fields_joiner(self.jp[0:1], document_schema_fields=['description', 'experienceRequirements']))
        assert len(joined[0]) == len(' '.join([self.jp[0]['description'], self.jp[0]['experienceRequirements']]))

    def test_html_cleaner(self):
        cleaned = list(html_cleaner(self.jp_with_html))
        s = "Overview:St. Michael's Inc. is seeking dedicated federal government " \
            "auditors from junior through the manager level to join our growing team for NGA."
        assert cleaned[0] ==  s

    def test_sentence_tokenizer(self):
        sentence_tokenized = list(sentence_tokenizer([self.jp[0]['description']]))
        assert sentence_tokenized[0] == self.nlp.sentence_tokenize(self.jp[0]['description'])

    def test_str_cleaner(self):
        cleaned = list(str_cleaner([self.jp[0]['description']]))
        assert cleaned[0] == self.nlp.clean_str(self.jp[0]['description'])

    def test_word_tokenizer(self):
        word_tokenized = list(word_tokenizer([self.jp[0]['description']]))
        assert word_tokenized[0] == self.nlp.word_tokenize(self.jp[0]['description'])

    def test_nlp_pipeline(self):
        def sentence_counter(generator):
            """count sentence for a document"""
            for doc in generator:
                yield len(doc)

        updated_fields_joiner = partial(fields_joiner, document_schema_fields=['description'])
        update_wrapper(updated_fields_joiner, fields_joiner)

        pipe1 = IterablePipeline(
                updated_fields_joiner,
                html_cleaner,
                sentence_tokenizer,
                str_cleaner,
                word_tokenizer,
                sentence_counter
                )
        pipe2 = IterablePipeline(
                updated_fields_joiner,
                html_cleaner,
                sentence_tokenizer,
                sentence_counter
                )

        assert pipe1.run(self.jp) == pipe2.run(self.jp)
        assert pipe1.description == [f.__doc__ for f in pipe1.functions]
        assert pipe1.description == ['Join selected fields. Each item is a document in json format.',
                                     'Remove html tags. Each item is a chunk of text.',
                                     'Tokenize sentences. Each item is a chunk of text.',
                                     'Remove punctuations, non-English letters, and lower case. Each item is a chunk of text.',
                                     'Tokenize words. Each item is a chunk of text.',
                                     'count sentence for a document']
