import json

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
        """
        """
        for line in generator:
            document = json.loads(line)
            #print(self._transform(document).split())
            yield self._transform(document).split()

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
