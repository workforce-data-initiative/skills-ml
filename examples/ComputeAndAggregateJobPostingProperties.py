"""Computing and aggregating job posting properties

To show job posting property computation and aggregation,
we calculate job posting counts by cleaned title, and upload
the resulting CSV to S3.

This is essentially a mini version of the Data@Work Research Hub.

To enable this example to be run with as few dependencies as possible, we use:
- a fake local s3 instance
- a sample of the Virginia Tech open job postings dataset
- only title cleaning and job counting.

To make this example a little bit more interesting, one could incorporate more
classes from the job_posting_properties.computers module, such as skill extractors or geocoders.

"""
import json
import logging
import urllib.request

from skills_ml.algorithms.job_posting_properties.computers import\
    TitleCleanPhaseOne, PostingIdPresent
from skills_ml.algorithms.job_posting_properties.aggregators import\
    aggregate_properties_for_quarter

from moto import mock_s3
import boto3
import s3fs
import unicodecsv as csv
import numpy

logging.basicConfig(level=logging.INFO)

VT_DATASET_URL = 'http://opendata.cs.vt.edu/dataset/ab0abac3-2293-4c9d-8d80-22d450254389/resource/9a810771-d6c9-43a8-93bd-144678cbdd4a/download/openjobs-jobpostings.mar-2016.json'


logging.info('Downloading sample Virginia Tech open jobs file')
response = urllib.request.urlopen(VT_DATASET_URL)
string = response.read().decode('utf-8')
logging.info('Download complete')
lines = string.split('\n')
logging.info('Found %s job posting lines', len(lines))

with mock_s3():
    client = boto3.resource('s3')
    client.create_bucket(Bucket='test-bucket')
    computed_properties_path = 's3://test-bucket/computed_properties'
    job_postings = []

    for line in lines:
        try:
            job_postings.append(json.loads(line))
        except ValueError:
            # Some rows in the dataset are not valid, just skip them
            logging.warning('Could not decode JSON')
            continue

    # create properties to be grouped on. In this case, we want to group on cleaned job title
    grouping_properties = [
        TitleCleanPhaseOne(path=computed_properties_path),
    ]
    # create properties to aggregate for each group
    aggregate_properties = [
        PostingIdPresent(path=computed_properties_path),
    ]

    # Regardless of their role in the final dataset, we need to compute
    # all properties from the dataset. Since the computed properties
    # partition their S3 caches by day, for optimum performance one
    # could parallelize each property's computation  by a day's worth of postings
    # But to keep it simple for this example, we are going to just runin a loop
    for cp in grouping_properties + aggregate_properties:
        logging.info('Computing property %s for %s job postings', cp, len(job_postings))
        cp.compute_on_collection(job_postings)

    # Now that the time consuming computation is done, we aggregate,
    # choosing an aggregate function for each aggregate column.
    # Here, the 'posting id present' property just emits the number 1,
    # so numpy.sum gives us a count of job postings
    # Many other properties, like skill counts, will commonly use
    # an aggregate function like 'most common'.
    # A selection is available in skills_ml.algorithms.aggregators.pandas
    logging.info('Aggregating properties')
    aggregate_path = aggregate_properties_for_quarter(
        quarter='2016Q1',
        grouping_properties=grouping_properties,
        aggregate_properties=aggregate_properties,
        aggregate_functions={'posting_id_present': [numpy.sum]},
        aggregations_path='s3://test-bucket/aggregated_properties',
        aggregation_name='title_state_counts'
    )

    s3 = s3fs.S3FileSystem()
    logging.info('Logging all rows in aggregate file')
    with s3.open(aggregate_path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            logging.info(row)
