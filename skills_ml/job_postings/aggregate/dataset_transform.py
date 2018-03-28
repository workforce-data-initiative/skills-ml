"""Track stats of job listing datasets, before and after transformation
into the common schema.
"""
from collections import Counter
from datetime import datetime
import json

import boto

from skills_utils.s3 import split_s3_path


def _compute_percentage(count, total):
    if total == 0:
        return 0
    else:
        return round(float(count)/total, 2)


def _compute_percentages(counts, total):
    return {
        field: _compute_percentage(count, total)
        for field, count in counts.items()
    }


class DatasetStatsCounter(object):
    """Accumulate data Dataset ETL statistics for a quarter
    to show presence and absence of different fields,
    and the total count of rows

    Args:
        dataset_id (string) A dataset id
        quarter (string) The quarter being analyzed
    """
    directory = 'quarterly'

    def __init__(self, dataset_id, quarter):
        self.dataset_id = dataset_id
        self.quarter = quarter
        self.stats = {
            'total': 0,
            'input_counts': Counter(),
            'output_counts': Counter(),
            'quarter': quarter,
        }

    def track(self, input_document, output_document):
        """Increment stats for a particular job listing,
        both pre-transformed and post-transformed
        """
        self.stats['total'] += 1
        for field in input_document:
            if input_document[field] != '':
                self.stats['input_counts'][field] += 1
        for field in output_document:
            if output_document[field] != '':
                self.stats['output_counts'][field] += 1

    def _compute_percentages(self):
        self.stats['input_percentages'] = _compute_percentages(
            self.stats['input_counts'],
            self.stats['total']
        )
        self.stats['output_percentages'] = _compute_percentages(
            self.stats['output_counts'],
            self.stats['total']
        )

    def save(self, s3_conn, s3_prefix):
        """Save stats to S3, including percentages
        Args:
            s3_conn (boto.Connection) - an s3 connection
            s3_prefix (str) s3 path (including bucket) to save dataset stats
        """
        bucket_name, prefix = split_s3_path(s3_prefix)
        bucket = s3_conn.get_bucket(bucket_name)
        self._compute_percentages()
        self.stats['last_updated'] = datetime.now().isoformat()
        key = boto.s3.key.Key(
            bucket=bucket,
            name='{}/{}/{}_{}'.format(
                prefix,
                self.directory,
                self.dataset_id,
                self.quarter
            )
        )
        key.set_contents_from_string(json.dumps(self.stats))

    @staticmethod
    def quarterly_posting_stats(s3_conn, stats_s3_path):
        bucket_name, prefix = split_s3_path(stats_s3_path)
        bucket = s3_conn.get_bucket(bucket_name)
        total = Counter()
        for key in bucket.list(
            prefix='{}/{}'.format(prefix, DatasetStatsCounter.directory)
        ):
            quarter = key.name[-6:]
            stats = json.loads(key.get_contents_as_string().decode('utf-8'))
            total[quarter] += stats['total']
        return total


class DatasetStatsAggregator(object):
    """Aggregate data Dataset ETL statistics up to the dataset level

    Args:
        dataset_id (string) A dataset id
        s3_conn (boto.Connection) an s3 connection
    """
    directory = 'dataset_summaries'

    def __init__(self, dataset_id, s3_conn):
        self.dataset_id = dataset_id
        self.s3_conn = s3_conn
        self.stats = {
            'total': 0,
            'input_counts': Counter(),
            'output_counts': Counter(),
            'quarters': {}
        }

    def _iterate_keys(self, s3_prefix):
        bucket_name, prefix = split_s3_path(s3_prefix)
        bucket = self.s3_conn.get_bucket(bucket_name)
        for key in bucket.list(
            prefix='{}/quarterly/{}_'.format(prefix, self.dataset_id)
        ):
            yield key

    def _accumulate_key(self, key):
        data = json.loads(key.get_contents_as_string().decode('utf-8'))
        self.stats['total'] += data['total']
        self.stats['input_counts'] += Counter(data['input_counts'])
        self.stats['output_counts'] += Counter(data['output_counts'])
        self.stats['quarters'][data['quarter']] = data

    def _compute_percentages(self):
        self.stats['input_percentages'] = _compute_percentages(
            self.stats['input_counts'],
            self.stats['total']
        )
        self.stats['output_percentages'] = _compute_percentages(
            self.stats['output_counts'],
            self.stats['total']
        )

    def _load(self, s3_prefix):
        bucket_name, prefix = split_s3_path(s3_prefix)
        bucket = self.s3_conn.get_bucket(bucket_name)
        key = self._key(bucket, prefix)
        self.stats = json.loads(key.get_contents_as_string().decode('utf-8'))

    def _save(self, s3_prefix):
        """Save stats to S3, including percentages
        """
        bucket_name, prefix = split_s3_path(s3_prefix)
        bucket = self.s3_conn.get_bucket(bucket_name)
        self._compute_percentages()
        self.stats['last_updated'] = datetime.now().isoformat()
        key = boto.s3.key.Key(
            bucket=bucket,
            name='{}/{}/{}.json'.format(
                prefix,
                self.directory,
                self.dataset_id
            )
        )
        key.set_contents_from_string(json.dumps(self.stats))

    def run(self, s3_prefix):
        """Compute stats and save them to s3

        Args:
            s3_prefix (str) s3 path (including bucket) to save dataset stats
        """
        for key in self._iterate_keys(s3_prefix):
            self._accumulate_key(key)
        self._save(s3_prefix)

    @staticmethod
    def partners(s3_conn, s3_prefix):
        partners_list = []
        bucket_name, prefix = split_s3_path(s3_prefix)
        bucket = s3_conn.get_bucket(bucket_name)
        for key in bucket.list(
            prefix='{}/{}'.format(prefix, DatasetStatsAggregator.directory)
        ):
            stats = json.loads(key.get_contents_as_string().decode('utf-8'))
            if stats['total'] > 0:
                partner_id = key.name.split('/')[-1].split('.')[0]
                partners_list.append(partner_id)
        return partners_list


class GlobalStatsAggregator(object):
    """Aggregate Dataset ETL statistics up to the global level

    Args:
        s3_conn (boto.Connection) an s3 connection
    """
    filename = 'summary.json'

    def __init__(self, s3_conn):
        self.s3_conn = s3_conn
        self.stats = {
            'total': 0,
            'output_counts': Counter(),
        }

    def _iterate_keys(self, s3_prefix):
        bucket_name, prefix = split_s3_path(s3_prefix)
        bucket = self.s3_conn.get_bucket(bucket_name)
        for key in bucket.list(prefix='{}/dataset_summaries/'.format(prefix)):
            yield key

    def _accumulate_key(self, key):
        data = json.loads(key.get_contents_as_string().decode('utf-8'))
        self.stats['total'] += data['total']
        self.stats['output_counts'] += Counter(data['output_counts'])

    def _compute_percentages(self):
        self.stats['output_percentages'] = _compute_percentages(
            self.stats['output_counts'],
            self.stats['total']
        )

    def _key(self, bucket, prefix):
        return boto.s3.key.Key(
            bucket=bucket,
            name='{}/{}'.format(prefix, self.filename)
        )

    def _save(self, s3_prefix):
        """Save stats to S3, including percentages
        """
        bucket_name, prefix = split_s3_path(s3_prefix)
        bucket = self.s3_conn.get_bucket(bucket_name)
        self._compute_percentages()
        self.stats['last_updated'] = datetime.now().isoformat()
        key = self._key(bucket, prefix)
        key.set_contents_from_string(json.dumps(self.stats))

    def _load(self, s3_prefix):
        bucket_name, prefix = split_s3_path(s3_prefix)
        bucket = self.s3_conn.get_bucket(bucket_name)
        key = self._key(bucket, prefix)
        self.stats = json.loads(key.get_contents_as_string().decode('utf-8'))

    def run(self, s3_prefix):
        """Compute stats and save them to s3

        Args:
            s3_prefix (str) s3 path (including bucket) to save dataset stats
        """
        for key in self._iterate_keys(s3_prefix):
            self._accumulate_key(key)
        self._save(s3_prefix)

    def saved_total(self, s3_prefix):
        self._load(s3_prefix)
        return self.stats['total']
