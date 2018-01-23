"""
Test the Virginia ETL transformer
"""
from skills_utils.testing import ImporterTest
from skills_ml.datasets.raw_job_postings import VirginiaTransformer

from moto import mock_s3_deprecated
import boto

from io import StringIO
import json


@mock_s3_deprecated
class VirginiaTest(ImporterTest):
    transformer_class = VirginiaTransformer

    sample_input_document = {
       "positionPeriod": {
          "startDate": "",
          "endDate": ""
       },
       "veteranCommitment": "",
       "datePosted": "2016-05-06",
       "hiringOrganization": {
          "organizationName": "My Org Name",
          "organizationTaxID": "",
          "url": "",
          "organizationCode": "123456",
          "organizationUnit": "",
          "additionalName": [],
          "location": "",
          "organizationDescription": "",
          "logo": "",
          "geo": {
             "latitude": None,
             "longitude": None
          }
       },
       "id": "4835a90ed38c6e4c224887d124d314b3",
       "salaryCurrency": "",
       "title": "F5 Technical Lead",
       "employmentType": [
          "Regular"
       ],
       "responsibilities": [],
       "normalizedTitle": {
          "onetName": "",
          "onetCode": ""
       },
       "numberOfOpenings": "1",
       "occupationalCategory": [],
       "incentiveCompensation": [],
       "specialCommitments": [],
       "jobBenefits": [],
       "qualifications": [],
       "baseSalary": {
          "salary": 0,
          "maxSalary": 0,
          "minSalary": 0,
          "medianSalary": 0
       },
       "dateExpires": "2016-06-06",
       "workHours": "",
       "experienceRequirements": [
          "96 month experience required",
          "Not Specified",
          "Job Description",
          "Typically has 8-10 years of related experience.",
          "College Degree or equivalent work experience."
       ],
       "url": "",
       "skills": [
          "Firewall, BlueCoat Proxy preferred"
       ],
       "jobDescription": "Responsible for network and network security...",
       "jobLocation": {
          "geo": {
             "latitude": None,
             "longitude": None
          },
          "address": {
             "postOfficeBox": "",
             "fullText": "",
             "countryCode": "",
             "locality": "Pittsburgh",
             "region": "PA",
             "countryName": "",
             "regionCode": "",
             "streetAddress": "",
             "postalCode": "",
             "extendedAddress": ""
          }
       },
       "educationRequirements": [
          "Bachelor's Degree"
       ]
    }

    def setUp(self):
        self.connection = boto.connect_s3()
        self.bucket_name = 'test'
        self.prefix = 'akey'
        self.transformer = self.transformer_class(
            bucket_name=self.bucket_name,
            prefix=self.prefix,
            partner_id='VA',
            s3_conn=self.connection,
        )

    def test_iterate(self):
        """Test that records from all files are properly returned or excluded
        according to the given date range"""
        bucket = self.connection.create_bucket(self.bucket_name)
        mock_data = {
            'fileone': [
                {'datePosted': '2014-12-15', 'dateExpires': '2015-01-15'},
                {'datePosted': '2014-11-15', 'dateExpires': '2014-12-15'},
                {'datePosted': '2015-01-15', 'dateExpires': '2015-02-15'},
            ],
            'filetwo': [
                {'datePosted': '2014-12-15', 'dateExpires': '2015-01-15'},
                {'datePosted': '2014-11-15', 'dateExpires': '2014-12-15'},
                {'datePosted': '2015-01-15', 'dateExpires': '2015-02-15'},
            ]
        }

        for keyname, rows in mock_data.items():
            key = boto.s3.key.Key(
                bucket=bucket,
                name='{}/{}'.format(self.prefix, keyname)
            )
            stream = StringIO()
            for row in rows:
                stream.write(json.dumps(row))
                stream.write('\n')
            stream.seek(0)
            key.set_contents_from_file(stream)

        postings = [
            posting
            for posting in self.transformer._iter_postings(quarter='2015Q1')
        ]
        assert len(postings) == 4

    def test_transform(self):
        """Test that the required fields are properly mapped
        from the input data"""
        transformed = self.transformer._transform(self.sample_input_document)
        assert transformed['title'] == 'F5 Technical Lead'
        assert transformed['jobLocation']['address']['addressLocality'] == \
            'Pittsburgh'
        assert transformed['jobLocation']['address']['addressRegion'] == 'PA'
        assert transformed['datePosted'] == '2016-05-06'
        assert transformed['validThrough'] == '2016-06-06T00:00:00'
        assert transformed['educationRequirements'] == "Bachelor's Degree"
        assert transformed['onet_soc_code'] == ''
        assert 'Responsible for' in transformed['description']
