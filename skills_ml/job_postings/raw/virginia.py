from datetime import datetime
import logging
import tempfile

from skills_utils.io import stream_json_file
from skills_utils.job_posting_import import JobPostingImportBase
from skills_utils.time import overlaps, quarter_to_daterange


def flatten(maybelist):
    if type(maybelist) is list:
        return ', '.join(maybelist)
    else:
        return maybelist


class VirginiaTransformer(JobPostingImportBase):
    DATE_FORMAT = '%Y-%m-%d'

    def __init__(self, bucket_name=None, prefix=None, **kwargs):
        super(VirginiaTransformer, self).__init__(**kwargs)
        self.bucket_name = bucket_name
        self.prefix = prefix

    def _iter_postings(self, quarter):
        logging.info("Finding Virginia postings for %s", quarter)
        quarter_start, quarter_end = quarter_to_daterange(quarter)
        bucket = self.s3_conn.get_bucket(self.bucket_name)
        keylist = list(bucket.list(prefix=self.prefix, delimiter=''))
        for key in keylist:
            if key.name.endswith('.cache.json'):
                continue

            logging.info("Processing key %s", key.name)
            with tempfile.NamedTemporaryFile() as local_file:
                key.get_contents_to_file(local_file)
                local_file.seek(0)
                for posting in stream_json_file(local_file):
                    if len(posting['datePosted']) == 0:
                        continue
                    listing_start = datetime.strptime(
                        posting['datePosted'],
                        self.DATE_FORMAT
                    )
                    if len(posting['dateExpires']) == 0:
                        listing_end = listing_start
                    else:
                        listing_end = datetime.strptime(
                            posting['dateExpires'],
                            self.DATE_FORMAT
                        )
                    if overlaps(
                        listing_start.date(),
                        listing_end.date(),
                        quarter_start,
                        quarter_end
                    ):
                        yield posting

    def _id(self, document):
        return document['id']

    def _transform(self, document):
        transformed = {
            "@context": "http://schema.org",
            "@type": "JobPosting",
        }
        basic_mappings = {
            'title': 'title',
            'description': 'jobDescription',
            'educationRequirements': 'educationRequirements',
            'employmentType': 'employmentType',
            'experienceRequirements': 'experienceRequirements',
            'incentiveCompensation': 'incentiveCompensation',
            'qualifications': 'qualifications',
            'occupationalCategory': 'occupationalCategory',
            'skills': 'skills',
            'id': 'id'
        }
        for target_key, source_key in basic_mappings.items():
            transformed[target_key] = flatten(document.get(source_key))

        if len(document['datePosted']) == 0:
            transformed['datePosted'] = None
        else:
            start = datetime.strptime(document['datePosted'], self.DATE_FORMAT)
            transformed['datePosted'] = start.date().isoformat()
        if len(document['dateExpires']) == 0:
            transformed['validThrough'] = None
        else:
            end = datetime.strptime(document['dateExpires'], self.DATE_FORMAT)
            transformed['validThrough'] = end.isoformat()

        transformed['jobLocation'] = {
            '@type': 'Place',
            'address': {
                '@type': 'PostalAddress',
                'addressLocality': document['jobLocation']['address']['locality'],
                'addressRegion': document['jobLocation']['address']['region'],
            }
        }
        transformed['baseSalary'] = {
            '@type': 'MonetaryAmount',
            'minValue': document['baseSalary']['minSalary'],
            'maxValue': document['baseSalary']['maxSalary'],
            'medianValue': document['baseSalary']['medianSalary'],
        }
        transformed['industry'] = document['hiringOrganization']['organizationCode']
        transformed['onet_soc_code'] = document['normalizedTitle']['onetCode']

        return transformed
