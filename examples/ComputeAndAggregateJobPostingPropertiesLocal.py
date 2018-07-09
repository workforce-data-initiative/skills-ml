"""Computing and aggregating job posting properties

To show job posting property computation and aggregation,
we calculate job posting counts by cleaned title, and upload
the resulting CSV to local file system.

This is essentially a mini version of the Data@Work Research Hub.

To enable this example to be run with as few dependencies as possible, we use:
- skills_ml.storage.FSStore() object with tempfile
- a sample of the Virginia Tech open job postings dataset
- only title cleaning and job counting.

To make this example a little bit more interesting, one could incorporate more
classes from the job_posting_properties.computers module, such as skill extractors or geocoders.

"""
import logging

from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.job_postings.computed_properties.computers import\
    TitleCleanPhaseOne, PostingIdPresent, Geography
from skills_ml.job_postings.geography_queriers.state import JobStateQuerier
from skills_ml.job_postings.computed_properties.aggregators import\
    aggregate_properties
from skills_ml.storage import FSStore
import unicodecsv as csv
import numpy
import os
import tempfile

logging.basicConfig(level=logging.INFO)


job_postings = list(JobPostingCollectionSample())

with tempfile.TemporaryDirectory() as tmpdir:
    computed_properties_path = os.path.join(tmpdir, 'computed_properties')
    storage = FSStore(computed_properties_path)

    # Create properties. In this example, we are going to both compute and aggregate,
    # but this is not necessary! Computation and aggregation are entirely decoupled.
    # So it's entirely valid to just compute a bunch of properties and then later
    # figure out how you want to aggregate them.
    # We are only introducing the 'grouping' and 'aggregate' semantics this early in the
    # script so as to avoid defining these properties twice in the same script.

    # create properties to be grouped on. In this case, we want to group on cleaned job title
    grouping_properties = [
        TitleCleanPhaseOne(storage=storage),
        Geography(geo_querier=JobStateQuerier(), storage=storage)
    ]
    # create properties to aggregate for each group
    aggregated_properties = [
        PostingIdPresent(storage=storage),
    ]

    # Regardless of their role in the final dataset, we need to compute
    # all properties from the dataset. Since the computed properties
    # partition their caches by day, for optimum performance one
    # could parallelize each property's computation by a day's worth of postings
    # But to keep it simple for this example, we are going to just runin a loop
    for cp in grouping_properties + aggregated_properties:
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
    aggregate_path = aggregate_properties(
        out_filename='2016Q1',
        grouping_properties=grouping_properties,
        aggregate_properties=aggregated_properties,
        aggregate_functions={'posting_id_present': [numpy.sum]},
        storage=storage,
        aggregation_name='title_state_counts'
    )

    logging.info('Logging all rows in aggregate file')
    with open(os.path.join(storage.path, aggregate_path), 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            logging.info(row)
