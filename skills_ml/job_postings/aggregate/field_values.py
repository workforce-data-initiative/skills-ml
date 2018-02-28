"""Track field value distribution of common schema job postings"""
from collections import Counter, defaultdict
from io import BytesIO
import unicodecsv as csv
import logging

import boto

from skills_utils.s3 import split_s3_path


class FieldValueCounter(object):
    """Accumulate field distribution statistics for common schema job postings

    Args:
        quarter (string) The quarter being analyzed
        field_values (list) each entry should be either:
            1. a field key
            2. a tuple, first value field key, second value function to fetch value or values from document
    """

    directory = 'field_values'

    def __init__(self, quarter, field_values):
        self.quarter = quarter
        self.field_values = field_values
        self.accumulator = defaultdict(Counter)

    def _accumulate_results(self, key, results):
        if isinstance(results, list):
            for result in results:
                self.accumulator[key][result] += 1
        else:
            self.accumulator[key][results] += 1

    def track(self, input_document):
        """Accumulate field values for a particular job listing"""
        for field in self.field_values:
            if isinstance(field, tuple):
                key, func = field
                self._accumulate_results(key, func(input_document))
            self._accumulate_results(field, input_document.get(field, None))

    def save(self, s3_conn, s3_prefix):
        """Save stats to S3, including percentages
        Args:
            s3_conn (boto.Connection) - an s3 connection
            s3_prefix (str) s3 path (including bucket) to save dataset stats
        """
        bucket_name, prefix = split_s3_path(s3_prefix)
        bucket = s3_conn.get_bucket(bucket_name)
        for field_name, counts in self.accumulator.items():
            output = BytesIO()
            writer = csv.writer(output)
            for value, count in counts.most_common():
                writer.writerow([value, count])

            key = boto.s3.key.Key(
                bucket=bucket,
                name='{}/{}/{}/{}.csv'.format(
                    prefix,
                    self.directory,
                    self.quarter,
                    field_name
                )
            )
            logging.info('Writing stats to %s', key)
            output.seek(0)
            key.set_contents_from_string(output.getvalue())
