from functools import reduce
from skills_ml.algorithms.string_cleaners.nlp import NLPTransforms, deep


class NLPPipeline(object):
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
    for document in generator:
        yield ' '.join([document.get(field, '') for field in document_schema_fields])


def html_cleaner(generator):
    for text in generator:
        yield deep(NLPTransforms().clean_html)(text)


def str_cleaner(generator):
    for item in generator:
        yield deep(NLPTransforms().clean_str)(item)


def word_tokenizer(generator):
    for text in generator:
        yield deep(NLPTransforms().word_tokenize)(text)


def sentence_tokenizer(generator):
    for raw_text in generator:
        yield NLPTransforms().sentence_tokenize(raw_text)

