import json
from algorithms.corpus_creators.basic import CorpusCreator

sample_input = [json.dumps({
    'test_field': 'We are looking for a person to fill this job'
})]

class ExampleCorpusCreator(CorpusCreator):
    def _transform(self, document):
        return document['test_field']

def test_raw_corpora():
    assert next(ExampleCorpusCreator().raw_corpora(sample_input)) == 'We are looking for a person to fill this job'

def test_array_corpora():
    assert next(ExampleCorpusCreator().array_corpora(sample_input)) == \
        ['We', 'are', 'looking', 'for', 'a', 'person', 'to', 'fill', 'this', 'job']
