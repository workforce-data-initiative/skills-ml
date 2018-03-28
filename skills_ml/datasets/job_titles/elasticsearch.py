"""
Index job title/occupation pairs in Elasticsearch.
"""
import uuid

from skills_utils.es import ElasticsearchIndexerBase


class JobTitlesMasterIndexer(ElasticsearchIndexerBase):
    """
    Args: job_title_generator (iterable). Each record is expected to be a dict
        with keys
        'Title' for the job title and
        'Original Title' for the occupation
    """
    def __init__(self, job_title_generator, alias_name, **kwargs):
        super(JobTitlesMasterIndexer, self).__init__(**kwargs)
        self.job_title_generator = job_title_generator
        self.alias_name = alias_name

    settings = {
        "number_of_shards": 1,
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
        }
    }
    mappings = {
        "titles": {
            "properties": {
                "occupation": {"type": "string", "analyzer": "english"},
                "jobtitle": {"type": "string", "analyzer": "english"}
            }
        }
    }

    data_keys = {
        'jobtitle': 'Title',
        'occupation': 'Original Title'
    }

    def generate_index_args(self, row, target_index):
        """Generate a job title/occupation record in a
        format that Elasticsearch can consume

        Args:
        row (dict) a data row representing one job/occupation pair
        target_index (string) the index to send the record

        Returns: dict
        """
        indexed_data = {
            key: row[value]
            for key, value in self.data_keys.items()
        }
        return {
            "_op_type": 'index',
            "_index": target_index,
            "_type": "titles",
            "_id": str(uuid.uuid4()),
            "_score": 1,
            "_source": indexed_data
        }

    def _iter_documents(self, target_index):
        """Take all available job titles and transform them into a format
        that Elasticsearch can index

        Args:
            target_index (string): the desired Elasticsearch index

        Returns: generator which yields dicts
        """
        return (
            self.generate_index_args(row, target_index)
            for row in self.job_title_generator
        )
