"""Aggregates job occupations
"""
import logging
import json
from skills_ml.algorithms.corpus_creators.basic import SimpleCorpusCreator
from skills_ml.algorithms.aggregators.geo import GeoAggregator


class GeoSocAggregator(GeoAggregator):
    """Aggregates job titles by geography

    Args:
        occupation_classifier (..occupation_classifiers.classifiers.Classifier,
            optional)
            a SOC code classifier,
            if absent will aggregate using the given SOC code
        corpus_creator (CorpusCreator, optional) an object that will transform
            a given common schema job posting into unstructured text for the
            occupation classifier. Defaults to SimpleCorpusCreator
    """
    def __init__(
        self,
        occupation_classifier=None,
        corpus_creator=None,
        *args,
        **kwargs
    ):
        super(GeoSocAggregator, self).__init__(*args, **kwargs)
        self.occupation_classifier = occupation_classifier
        self.corpus_creator = corpus_creator or SimpleCorpusCreator()
        self.job_key_name = 'soc_code'

    def process_postings(self, job_postings):
        """
        Computes the SOC/CBSA distribution of the given job postings
        Args:
            job_postings (iterable) Job postings, each in common schema format

        When complete, the aggregators in self.job_aggregators will be updated
        with data from the job postings
        """
        for i, line in enumerate(job_postings):
            job_posting = json.loads(line)
            if self.occupation_classifier:
                soc_code, _ = \
                    self.occupation_classifier.classify(
                        self.corpus_creator._transform(job_posting)
                    )
            else:
                soc_code = job_posting.get('onet_soc_code', '99-9999.00')
            geography_hits = self.geo_querier.query(line)

            for aggregator in self.job_aggregators.values():
                aggregator.accumulate(
                    job_posting=job_posting,
                    job_key=soc_code,
                    groups=(geography_hits,)
                )
            if i % 1000 == 0:
                logging.info('Aggregated %s job postings', i)
