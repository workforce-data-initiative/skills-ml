from functools import reduce
from skills_ml.algorithms.string_cleaners.nlp import NLPTransforms, deep


class NLPPipeline(object):
    """A simple nlp preprocessing pipeline.

    Attributes:
        functions (func): a series of generator functions that takes another generator as input

    """
    def __init__(self, *functions):
        self.functions = list(functions)

    @property
    def compose(self):
        return reduce(lambda f, g: lambda x: g(f(x)), self.functions, lambda x: x)

    def run(self, generator):
        return list(self.compose(generator))

    @property
    def description(self):
        return [f.__doc__ for f in self.functions]


def fields_joiner(generator, document_schema_fields=['description','experienceRequirements', 'qualifications', 'skills']):
    """join selected fields"""
    for document in generator:
        yield ' '.join([document.get(field, '') for field in document_schema_fields])


def html_cleaner(generator):
    """remove html tags"""
    for text in generator:
        yield deep(NLPTransforms().clean_html)(text)


def str_cleaner(generator):
    """remove punctuations, non-English letters, and lower case"""
    for item in generator:
        yield deep(NLPTransforms().clean_str)(item)


def word_tokenizer(generator):
    """tokenize words"""
    for text in generator:
        yield deep(NLPTransforms().word_tokenize)(text)


def sentence_tokenizer(generator):
    """tokenize sentences"""
    for raw_text in generator:
        yield NLPTransforms().sentence_tokenize(raw_text)
