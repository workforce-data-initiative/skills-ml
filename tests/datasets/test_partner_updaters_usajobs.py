from skills_ml.datasets.partner_updaters import USAJobsUpdater
import httpretty
import json

FAKE_KEY = 'faaaaake'
FAKE_EMAIL = 'fake@email.com'

PAGE_1 = {
    'SearchResult': {
        'UserArea': {
            'NumberOfPages': 2,
        },
        'SearchResultItems': [
            {'MatchedObjectId': 5, 'MatchedObjectDescriptor': {'k': 'v'}},
            {'MatchedObjectId': 6, 'MatchedObjectDescriptor': {'k': 'v2'}},
        ],
        'SearchResultCount': 2,
        'SearchResultCountAll': 4,
    }
}

PAGE_2 = {
    'SearchResult': {
        'UserArea': {
            'NumberOfPages': 2,
        },
        'SearchResultItems': [
            {'MatchedObjectId': 6, 'MatchedObjectDescriptor': {'k': 'v2'}},
            {'MatchedObjectId': 7, 'MatchedObjectDescriptor': {'k': 'v3'}},
        ],
        'SearchResultCount': 2,
        'SearchResultCountAll': 4,
    }
}


def callback(request, uri, headers):
    if 'Page=1' in uri:
        body = json.dumps(PAGE_1)
    else:
        body = json.dumps(PAGE_2)
    return (200, headers, body)


@httpretty.activate
def test_usa_jobs_updater_deduplicated_items():
    httpretty.register_uri(
        httpretty.GET,
        USAJobsUpdater.base_url,
        body=callback
    )
    updater = USAJobsUpdater(
        auth_key=FAKE_KEY,
        key_email=FAKE_EMAIL,
    )
    postings = updater.deduplicated_postings()
    assert len(postings.keys()) == 3
    assert sorted(postings.keys()) == [5, 6, 7]
    assert postings == {
        5: {'k': 'v'},
        6: {'k': 'v2'},
        7: {'k': 'v3'},
    }
