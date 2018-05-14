"""Import USAJobs postings into the Open Skills common schema"""
from datetime import datetime
import logging
import json

from skills_utils.job_posting_import import JobPostingImportBase
from skills_utils.time import overlaps, quarter_to_daterange


class USAJobsTransformer(JobPostingImportBase):
    DATE_FORMAT = '%Y-%m-%d'

    def __init__(self, bucket_name=None, prefix=None, **kwargs):
        super(USAJobsTransformer, self).__init__(**kwargs)
        self.bucket_name = bucket_name
        self.prefix = prefix

    def _iter_postings(self, quarter):
        """Iterate through raw postings for a given quarter

        Args:
            quarter (string): A quarter (in format 2015Q1)

        Yields:
            Untransformed job postings (dict)
        """
        logging.info("Finding USAJobs postings for %s", quarter)
        quarter_start, quarter_end = quarter_to_daterange(quarter)
        bucket = self.s3_conn.get_bucket(self.bucket_name)
        full_prefix = self.prefix + '/' + quarter
        keylist = list(bucket.list(prefix=full_prefix, delimiter=''))
        for key in keylist:
            logging.info("Processing key %s", key.name)
            contents = key.get_contents_as_string()
            posting = json.loads(contents.decode('utf-8'))
            posting['id'] = key.name.split('.')[-2]
            if len(posting['PositionStartDate']) == 0:
                continue
            listing_start = datetime.strptime(
                posting['PositionStartDate'],
                self.DATE_FORMAT
            )
            if len(posting['PositionEndDate']) == 0:
                listing_end = listing_start
            else:
                listing_end = datetime.strptime(
                    posting['PositionEndDate'],
                    self.DATE_FORMAT
                )
            if overlaps(
                listing_start.date(),
                listing_end.date(),
                quarter_start,
                quarter_end
            ):
                yield posting
            else:
                logging.warning(
                    'Posting %s does not overlap with quarter %s',
                    posting['id'],
                    quarter
                )

    def _id(self, document):
        """Given a raw job posting, return its vendor-specific unique id

        Args:
            document (dict) a job posting

        Returns: (string) an identifier
        """
        return document['id']

    def _transform(self, document):
        """Given a raw job posting, transform it into the common schema

        Args:
            document (dict) a job posting

        Returns: (dict) the job posting in common schema format
        """
        transformed = {
            "@context": "http://schema.org",
            "@type": "JobPosting",
        }
        basic_mappings = {
            'title': 'PositionTitle',
            'qualifications': 'QualificationSummary',
            'url': 'PositionURI',
        }
        for target_key, source_key in basic_mappings.items():
            transformed[target_key] = document.get(source_key)

        # many of the fields we want are in UserArea->Details
        # sadly most of these never seem to show up in real data,
        # but they are mentioned in the API docs so they are worth checking for
        user_details = document.get('UserArea', {}).get('Details', {})
        transformed['description'] = user_details.get('JobSummary', None)
        transformed['educationRequirements'] = \
            user_details.get('Education', None)
        transformed['responsibilities'] = user_details.get('MajorDuties', None)
        transformed['experienceRequirements'] = \
            user_details.get('Requirements', None)
        transformed['jobBenefits'] = user_details.get('Benefits', None)

        # employment type, salary, and location are stored in lists;
        # pick the first one
        position_schedules = document.get('PositionSchedule', [])
        if len(position_schedules) > 0:
            transformed['employmentType'] = \
                position_schedules[0].get('Name', None)

        remuneration = document.get('PositionRemuneration', [])
        if len(remuneration) > 0:
            transformed['baseSalary'] = {
                '@type': 'MonetaryAmount',
                'minValue': float(remuneration[0].get('MinimumRange', None)),
                'maxValue': float(remuneration[0].get('MaximumRange', None))
            }

        locations = document.get('PositionLocation', [])
        if len(locations) > 0:
            transformed['jobLocation'] = {
                '@type': 'Place',
                'address': {
                    '@type': 'PostalAddress',
                    'addressLocality': locations[0].get('CityName', ''),
                    'addressRegion': locations[0].get('CountrySubDivisionCode', ''),
                    'addressCountry': locations[0].get('CountryCode', ''),
                }
            }

        # both organization and the department within the org. are defined
        transformed['hiringOrganization'] = {
            '@type': 'Organization',
            'name': document.get('OrganizationName')
        }
        department_name = document.get('DepartmentName', None)
        if department_name:
            transformed['hiringOrganization']['department'] = {
                '@type': 'Organization',
                'name': department_name
            }

        if not document['PositionStartDate']:
            transformed['datePosted'] = None
        else:
            start = datetime.strptime(
                document['PositionStartDate'],
                self.DATE_FORMAT
            )
            transformed['datePosted'] = start.date().isoformat()
        if not document['PositionEndDate']:
            transformed['validThrough'] = None
        else:
            end = datetime.strptime(
                document['PositionEndDate'],
                self.DATE_FORMAT
            )
            transformed['validThrough'] = end.isoformat()

        return transformed
