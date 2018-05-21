from skills_ml.job_postings.corpora.basic import CorpusCreator

sample_input = [{
    'description': '<html><h1>We are looking for engineer\n\n</h1></html>',
    'test_field': 'We are looking for a person to fill this job'
}]


class ExampleCorpusCreator(CorpusCreator):
    def _transform(self, document):
        return document['test_field']

def test_clean():
    corpus = CorpusCreator(sample_input, ['description'])
    assert next(iter(corpus))['description'] == 'We are looking for engineer'

def test_metadata():
    corpus = CorpusCreator()
    assert corpus.metadata['corpus_creator'] == 'skills_ml.job_postings.corpora.basic.CorpusCreator'
