import json
from skills_ml.job_postings.corpora.basic import CorpusCreator

sample_input = [json.dumps({
    'description': '<html><h1>We are looking for engineer\n\n</h1></html>',
    'test_field': 'We are looking for a person to fill this job'
})]

def test_raw_corpora():
    assert next(iter(CorpusCreator(sample_input, ['test_field'], raw=True))) == 'We are looking for a person to fill this job'

def test_array_corpora():
    assert next(iter(CorpusCreator(sample_input, ['test_field'], raw=True, tokenize=True))) == \
        ['We', 'are', 'looking', 'for', 'a', 'person', 'to', 'fill', 'this', 'job']

def test_clean():
    corpus = CorpusCreator(sample_input, ['description'])
    assert next(iter(corpus)) == 'we are looking for engineer'

def test_metadata():
    corpus = CorpusCreator()
    assert corpus.metadata['corpus_creator'] == 'skills_ml.job_postings.corpora.basic.CorpusCreator'
