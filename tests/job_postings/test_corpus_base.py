import json
from skills_ml.job_postings.corpora.basic import CorpusCreator

sample_input = [json.dumps({
    'description': '<html><h1>We are looking for engineer\n\n</h1></html>',
    'test_field': 'We are looking for a person to fill this job'
})]

class ExampleCorpusCreator(CorpusCreator):
    def _transform(self, document):
        return document['test_field']

def test_raw_corpora():
    assert next(ExampleCorpusCreator().raw_corpora(sample_input)) == 'We are looking for a person to fill this job'

def test_array_corpora():
    assert next(ExampleCorpusCreator().tokenize_corpora(sample_input)) == \
        ['We', 'are', 'looking', 'for', 'a', 'person', 'to', 'fill', 'this', 'job']

def test_clean():
    corpus = CorpusCreator()
    assert corpus._clean(json.loads(sample_input[0]))['description'] == 'We are looking for engineer'

def test_metadata():
	corpus = CorpusCreator()
	assert corpus.metadata['corpus_creator'] == 'skills_ml.job_postings.corpora.basic.CorpusCreator'
