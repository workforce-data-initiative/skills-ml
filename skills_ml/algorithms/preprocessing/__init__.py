from functools import reduce

from skills_ml.algorithms.string_cleaners.nlp import NLPTransforms, deep


class NLPPipeline(object):
    """A simple nlp preprocessing pipeline.

    This class will compose preprocessing functions together to be passed to different stages(training/prediction)
    to assert the same preprocessing procedrues.

    Example:
        ```python
        jp = JobPostingCollectionSample()
        pipe = NLPPipeline(
            partial(fields_joiner, document_schema_fields=['description']),
            html_cleaner,
            sentence_tokenizer,
            str_cleaner,
            word_tokenizer
        )
        result = pipe.run(jp)
        ```

    Attributes:
        functions (func): a series of generator functions that takes another generator as input

    """
    def __init__(self, *functions):
        self.functions = list(functions)

    @property
    def compose(self):
        """compose functions

        Returns:
            generator: a generator objet which is not materialized yet
        """
        return reduce(lambda f, g: lambda x: g(f(x)), self.functions, lambda x: x)

    def run(self, generator):
        """materialize the generator object

        Returns:
            list: a list of itmes after apply all the functions on them
        """
        return list(self.compose(generator))

    @property
    def description(self):
        """pipeline description"""
        return [f.__doc__ for f in self.functions]


def fields_joiner(generator, document_schema_fields=['description','experienceRequirements', 'qualifications', 'skills']):
    """Join selected fields. Each item is a document in json format."""
    for document in generator:
        yield ' '.join([document.get(field, '') for field in document_schema_fields])


def html_cleaner(generator):
    """Remove html tags. Each item is a chunk of text."""
    for text in generator:
        yield deep(NLPTransforms().clean_html)(text)


def str_cleaner(generator):
    """Remove punctuations, non-English letters, and lower case. Each item is a chunk of text."""
    for item in generator:
        yield deep(NLPTransforms().clean_str)(item)


def word_tokenizer(generator):
    """Tokenize words. Each item is a chunk of text."""
    for text in generator:
        yield deep(NLPTransforms().word_tokenize)(text)


def sentence_tokenizer(generator):
    """Tokenize sentences. Each item is a chunk of text."""
    for raw_text in generator:
        yield NLPTransforms().sentence_tokenize(raw_text)
