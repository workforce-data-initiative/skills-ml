import json
from random import randint
from utils.nlp import NLPTransforms


class CorpusCreator(object):
    """
        A base class for objects that convert common schema
        job listings into a corpus suitable for use by
        machine learning algorithms.

        Subclasses should implement _transform(document)
    """
    def __init__(self):
        self.nlp = NLPTransforms()

    def raw_corpora(self, generator):
        """Transforms job listings into corpus format

        Args:
            generator: an iterable that generates JSON strings.
                Each string is expected to represent a job listing
                conforming to the common schema
                See sample_job_listing.json for an example of this schema

        Yields:
            (string) The next job listing transformed into corpus format
        """
        for line in generator:
            document = json.loads(line)
            yield self._transform(document)

    def array_corpora(self, generator):
        """Transforms job listings into corpus format for gensim's doc2vec

        Args:
            generator: an iterable that generates an array of words(strings).
                Each array is expected to represent a job listing(a doc)
                including fields of interests
        Yields:
            (list) The next job listing transformed into gensim's doc2vec

        """
        for line in generator:
            document = json.loads(line)
            yield self._transform(document).split()

    def label_corpora(self, generator):
        """Extract job label(category) from job listings and transfrom into corpus format

        Args:
            generator: an iterable that generates a list of job label (strings).

        Yields:
            (string) The next job label transform into corpus format
        """
        for line in generator:
            document = json.loads(line)
            yield str(randint(0,23))

class SimpleCorpusCreator(CorpusCreator):
    """
        An object that transforms job listing documents by picking
        important schema fields and returns them as one large lowercased string
    """
    document_schema_fields = [
        'description',
        'experienceRequirements',
        'qualifications',
        'skills'
    ]

    join_spaces = ' '.join

    def _transform(self, document):
        return self.join_spaces([
            self.nlp.lowercase_strip_punc(document[field])
            for field in self.document_schema_fields
        ])

class GensimCorpusCreator(CorpusCreator):
    """
        An object that transforms job listing documents by picking
        important schema fields and returns them as one large cleaned array of words
    """
    document_schema_fields = [
        'description',
        'experienceRequirements',
        'qualifications',
        'skills'
    ]
    join_spaces = ' '.join

    def _transform(self, document):
        return self.join_spaces([
            self.nlp.clean_str(document[field])
            for field in self.document_schema_fields
        ])

class JobCategoryCorpusCreator(CorpusCreator):
    """
        An object that extract the label of each job listing document which could be onet soc code or
        occupationalCategory and returns them as a lowercased string
    """
    document_schema_fields = [
        'occupationalCategory']

    def _transform(self, document):
        return self.join_spaces([
            self.nlp.lowercase_strip_punc(document[field])
            for field in self.document_schema_fields
        ])
