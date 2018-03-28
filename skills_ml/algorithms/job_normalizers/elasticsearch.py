"""
Indexes job postings for job title normalization
"""
import uuid
import json

from skills_utils.es import ElasticsearchIndexerBase


class NormalizeTopNIndexer(ElasticsearchIndexerBase):
    """Creates an index that stores data for job title normalization.
    
    Depends on a previously created index with job titles and occupations.

    Queries the job title/occupation index for
    1. job titles or occupations that match the job description
    2. Occupation matches

    The top three results are indexed.

    Args:
        quarter (string) the quarter from which to retrieve job postings
        job_postings_generator (iterable) an iterable of job postings
        job_title_index (string) The name of an already existing job title/occupation index
    """
    settings = {
        "number_of_shards": 1,
        "index.codec": "best_compression",
        "analysis": {
            "analyzer": {
                "english": {
                    "tokenizer":  "standard",
                    "filter": [
                        "english_possessive_stemmer",
                        "lowercase",
                        "english_stop",
                        "english_stemmer"
                    ]
                }
            },
            "filter": {
                "english_stop": {
                    "type":       "stop",
                    "stopwords":  "_english_"
                },
                "english_stemmer": {
                    "type":       "stemmer",
                    "language":   "english"
                },
                "english_possessive_stemmer": {
                    "type":       "stemmer",
                    "language":   "possessive_english"
                }
            }
        },
    }
    mappings = {
        "titles": {
            "_all": {"enabled": False},
            "properties": {
                "occupation": {"type": "string"},
                "jobtitle": {"type": "string"},
                "jobdesc": {"type": "string", "analyzer": "english"}
            }
        }
    }

    data_keys = {
        'occupation': 'occupationalCategory',
        'jobdesc': 'description',
        'jobtitle': 'title'
    }

    def __init__(
        self,
        quarter,
        job_postings_generator,
        job_titles_index,
        alias_name,
        **kwargs
    ):
        super(NormalizeTopNIndexer, self).__init__(**kwargs)
        self.quarter = quarter
        self.job_postings_generator = job_postings_generator
        self.job_titles_index = job_titles_index
        self.alias_name = alias_name

    def retrieve_top_titles(self, jobdesc, occupation):
        """Retrieves job titles from Elasticsearch that match the arguments

        Queries the job title/occupation index for:
        1. job titles or occupations that match the job description
        2. Occupation matches
        The top three results are returned

        Args:
            jobdesc (string) A job description
            occupation (string) An occupation

        Returns: A list of strings, each a matching job titles
        """
        body = {
            "size": 3,
            "_source": ["jobtitle"],
            "query": {
                "bool": {
                    "should": [
                        {"multi_match": {
                            "fields": ["jobtitle", "occupation"],
                            "query": jobdesc[:1000],
                        }},
                        {"match": {"occupation": occupation}}
                    ]
                }
            }
        }
        response = self.es_client.search(index=self.job_titles_index, body=body)
        results = response['hits']['hits']
        return [row['_source']['jobtitle'][0] for row in results]

    def generate_index_args(self, posting, target_index):
        """Generate indexable job posting records

        Queries the Elasticsearch job title index for the
        closest job titles matching the description and occupation, and returns
        the top three results

        Args:
        posting (dict) A job posting
        target_index (string) The desired Elasticseach index

        Yields: dicts representing indexable records
        """
        indexed_data = {
            key: posting[value]
            for key, value in self.data_keys.items()
        }
        indexed_data['quarters'] = [self.quarter]
        for matched_title in self.retrieve_top_titles(
            indexed_data['jobdesc'],
            indexed_data['occupation']
        ):
            indexed_data['canonicaltitle'] = matched_title
            yield {
                "_op_type": 'index',
                "_index": target_index,
                "_type": "titles",
                "_id": str(uuid.uuid4()),
                "_score": 1,
                "_source": indexed_data
            }

    def _iter_documents(self, target_index):
        """Take all available job postings and generates documents
        that Elasticsearch can index

        Args:
            target_index (string): the desired Elasticsearch index

        Returns: generator which yields dicts
        """
        return (
            data
            for posting in self.job_postings_generator(self.s3_conn, self.quarter)
            for data in self.generate_index_args(json.loads(posting), target_index)
        )
