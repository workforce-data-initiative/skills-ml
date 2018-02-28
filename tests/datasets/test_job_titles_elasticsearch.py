from skills_utils.es import basic_client
from skills_ml.datasets.job_titles.elasticsearch import JobTitlesMasterIndexer


def test_index_titles():
    input_data = [
        {'Title': 'Job 1', 'Original Title': 'Occupation 1'},
        {'Title': 'Job 2', 'Original Title': 'Occupation 2'},
        {'Title': 'Job 3', 'Original Title': 'Occupation 3'},
    ]

    indexer = JobTitlesMasterIndexer(
        job_title_generator=input_data,
        alias_name='stuff',
        s3_conn=None,
        es_client=basic_client()
    )

    index = 'stuff'

    documents = [document for document in indexer._iter_documents(target_index=index)]

    assert len(documents) == 3

    for document in documents:
        assert 'Occupation ' in document['_source']['occupation']
        assert 'Job ' in document['_source']['jobtitle']
        assert document['_index'] == index
        assert document['_id']
