from random import randint
from skills_ml.algorithms.nlp import clean_html, clean_str, lowercase_strip_punc, word_tokenize, sentence_tokenize, section_extract, strip_bullets_from_line
from gensim.models.doc2vec import TaggedDocument
from skills_utils.common import safe_get


class CorpusCreator(object):
    """
        A base class for objects that convert common schema
        job listings into a corpus in documnet level suitable for use by
        machine learning algorithms or specific tasks.

    Example:
    ```python
    from skills_ml.job_postings.common_schema import JobPostingCollectionSample
    from skills_ml.job_postings.corpora.basic import CorpusCreator

    job_postings_generator = JobPostingCollectionSample()

    # Default will include all the cleaned job postings
    corpus = CorpusCreator(job_postings_generator)

    # For getting a the raw job postings without any cleaning
    corpus = CorpusCreator(job_postings_generator, raw=True)
    ```


    Attributes:
        job_posting_generator (generator):  an iterable that generates JSON strings.
                                Each string is expected to represent a job listing
                                conforming to the common schema
                                See sample_job_listing.json for an example of this schema
        document_schema_fields (list): an list of schema fields to be included
        raw (bool): a flag whether to return the raw documents or transformed documents

    Yield:
        (dict): a dictinary only with selected fields as keys and corresponding raw/cleaned value
    """
    def __init__(self, job_posting_generator=None, document_schema_fields=['description','experienceRequirements', 'qualifications', 'skills'],
                 raw=False):
        self.job_posting_generator = job_posting_generator
        self.raw = raw
        self.document_schema_fields = document_schema_fields
        self.join_spaces = ' '.join
        self.key = ['onet_soc_code']

    @property
    def metadata(self):
        meta_dict = {'corpus_creator': ".".join([self.__module__ , self.__class__.__name__])}
        if self.job_posting_generator:
            meta_dict.update(self.job_posting_generator.metadata)
        return meta_dict

    def _clean(self, document):
        for f in self.document_schema_fields:
            try:
                cleaned = clean_html(document[f]).replace('\n','')
                cleaned = " ".join(cleaned.split())
                document[f] = cleaned
            except KeyError:
                pass
        return document

    def _transform(self, document):
        if self.raw:
            return self._join(document)
        else:
            return self._clean(document)

    def _join(self, document):
        return self.join_spaces([
            document.get(field, '') for field in self.document_schema_fields
        ])

    def __iter__(self):
        for document in self.job_posting_generator:
            document = {key: document[key] for key in self.document_schema_fields}
            yield self._transform(document)


class SimpleCorpusCreator(CorpusCreator):
    """
        An object that transforms job listing documents by picking
        important schema fields and returns them as one large lowercased string
    """
    def _clean(self, document):
        return self.join_spaces([
            lowercase_strip_punc(document.get(field, ''))
            for field in self.document_schema_fields
        ])


class Doc2VecGensimCorpusCreator(CorpusCreator):
    """Corpus for training Gensim Doc2Vec
    An object that transforms job listing documents by picking
    important schema fields and yields them as one large cleaned array of words

    Example:
    ```python

    from skills_ml.job_postings.common_schema import JobPostingCollectionSample
    from skills_ml.job_postings.corpora.basic import Doc2VecGensimCorpusCreator

    job_postings_generator = JobPostingCollectionSample()

    corpus = Doc2VecGensimCorpusCreator(job_postings_generator)

    Attributes:
        job_posting_generator (generator): a job posting generator
        document_schema_fields (list): an list of schema fields to be included
    """
    def __init__(self, job_posting_generator, document_schema_fields=['description','experienceRequirements', 'qualifications', 'skills'], *args, **kwargs):
        super().__init__(job_posting_generator, document_schema_fields, *args, **kwargs)
        self.lookup = {}
        self.k = 0 if not self.lookup else max(self.lookup.keys()) + 1

    def _clean(self, document):
        return self.join_spaces([
            clean_str(document[field])
            for field in self.document_schema_fields
        ])

    def _transform(self, document):
        words = self._clean(document).split()
        tag = [self.k]
        return TaggedDocument(words, tag)

    def __iter__(self):
        for document in self.job_posting_generator:
            self.lookup[self.k] = safe_get(document, *self.key)
            yield self._transform(document)
            self.k += 1


class Word2VecGensimCorpusCreator(CorpusCreator):
    """
        An object that transforms job listing documents by picking
        important schema fields and yields them as one large cleaned array of words
    """
    def __init__(self, job_posting_generator, document_schema_fields=['description','experienceRequirements', 'qualifications', 'skills'], *args, **kwargs):
        super().__init__(job_posting_generator, document_schema_fields, *args, **kwargs)

    def _clean(self, document):
        return self.join_spaces([
            clean_str(document[field])
            for field in self.document_schema_fields
        ])

    def _transform(self, document):
        if self.raw:
            return [word_tokenize(s) for s in sentence_tokenize(self._join(document))]
        else:
            return [word_tokenize(s) for s in sentence_tokenize(self._clean(document))]

    def __iter__(self):
        for document in self.job_posting_generator:
            document = {key: document[key] for key in self.document_schema_fields}
            sentences = self._transform(document)
            for sentence in sentences:
                yield sentence

class JobCategoryCorpusCreator(CorpusCreator):
    """
        An object that extract the label of each job listing document which could be onet soc code or
        occupationalCategory and yields them as a lowercased string
    """
    document_schema_fields = [
        'occupationalCategory']

    def _transform(self, document):
        return self.join_spaces([
            lowercase_strip_punc(document[field])
            for field in self.document_schema_fields
        ])


class SectionExtractWord2VecCorpusCreator(Word2VecGensimCorpusCreator):
    """Only return the contents of the configured section headers.

    Heavily utilizes skills_ml.algorithms.nlp.section_extract.
    For more detail on how to define 'sections', refer to its docstring.
    """
    def __init__(self, section_regex, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.section_regex = section_regex

    def _transform(self, document):
        lines_from_section = section_extract(self.section_regex, document['description'])
        return [word_tokenize(clean_str(strip_bullets_from_line(line.text))) for line in lines_from_section]


class RawCorpusCreator(CorpusCreator):
    """
        An object that yields the joined raw string of job posting
    """
    def __init__(self, job_posting_generator, document_schema_fields=['description','experienceRequirements', 'qualifications', 'skills']):
        super().__init__(job_posting_generator, document_schema_fields)

    def _transform(self, document):
        return self.join_spaces([document[field] for field in self.document_schema_fields])
