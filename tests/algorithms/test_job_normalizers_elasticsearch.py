import httpretty
import json
import re

from skills_utils.es import basic_client

from skills_ml.algorithms.job_normalizers.elasticsearch import NormalizeTopNIndexer


def mock_job_posting_generator(postings):
    return lambda s3_conn, quarter: (json.dumps(post) for post in postings)


@httpretty.activate
def test_normalize_topn():
    generator = mock_job_posting_generator((
        {
            'occupationalCategory': 'Line Cooks, Chefs',
            'description': 'A food job description',
            'title': 'Food Truck Sous Chef'
        },
        {
            'occupationalCategory': 'Actors',
            'description': 'An actor job description',
            'title': 'Broadway Star'
        },
    ))
    indexer = NormalizeTopNIndexer(
        s3_conn=None,
        es_client=basic_client(),
        job_titles_index='stuff',
        alias_name='otherstuff',
        quarter='2014Q1',
        job_postings_generator=generator
    )

    # TODO: abstract the ES mocking to a module
    # This means that the titles endpoint will say that all input job titles
    # match best with 'Janitor'
    mock_result = {
        'hits': {
            'hits': [
                {'_source': {'jobtitle': ['Janitor']}}
            ]
        }
    }
    httpretty.register_uri(
        httpretty.GET,
        re.compile('http://localhost:9200/stuff/_search'),
        body=json.dumps(mock_result),
        content_type='application/json'
    )

    index = 'stuff'

    documents = [document for document in indexer._iter_documents(target_index=index)]

    assert len(documents) == 2

    for document in documents:
        assert document['_source']['canonicaltitle'] == 'Janitor'
        assert document['_source']['quarters'] == ['2014Q1']
        assert document['_source']['occupation'] in ['Line Cooks, Chefs', 'Actors']
        assert document['_source']['jobtitle'] in ['Food Truck Sous Chef', 'Broadway Star']
        assert document['_index'] == index
        assert document['_id']
