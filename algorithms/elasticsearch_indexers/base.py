"""
Base class for Elasticsearch indexers

Subclasses implement the index setting definition and transformation of data,
The base class handles index management and bulk indexing with ES
"""
import logging
from utils.es import get_index_from_alias, zero_downtime_index
from elasticsearch.helpers import streaming_bulk
from config import config


class ElasticsearchIndexerBase(object):
    def __init__(self, s3_conn, es_client, app_config=None):
        """
        Args:
            s3_conn - a boto s3 connection
            es_client - an Elasticsearch indices client
            config (dict) config to override YAML file config
        """
        self.s3_conn = s3_conn
        self.es_client = es_client
        self.config = app_config or config

    def index_config(self):
        """Combines setting and mapping config into a full index configuration
        Returns: dict
        """
        return {
            'settings': self.settings,
            'mappings': self.mappings
        }

    def replace(self):
        """Replace index with a new one
        zero_downtime_index for safety and rollback
        """
        with zero_downtime_index(self.alias_name, self.index_config()) as target_index:
            self.index_all(target_index)

    def append(self):
        """Index documents onto an existing index"""
        target_index = get_index_from_alias(self.alias_name)
        if not target_index:
            self.replace()
        else:
            self.index_all(target_index)

    def index_all(self, index_name):
        """Index all available documents, using streaming_bulk for speed
        Args:

        index_name (string): The index
        """
        oks = 0
        notoks = 0
        for ok, item in streaming_bulk(
            self.es_client,
            self._iter_documents(index_name)
        ):
            if ok:
                oks += 1
            else:
                notoks += 1
        logging.info(
            "Import results: %d ok, %d not ok",
            oks,
            notoks
        )
