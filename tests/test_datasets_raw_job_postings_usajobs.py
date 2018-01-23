"""
Test the USAJobs ETL transformer
"""
from skills_utils.testing import ImporterTest
from skills_ml.datasets.raw_job_postings import USAJobsTransformer

from moto import mock_s3_deprecated
import boto
import uuid

from io import StringIO
import json


@mock_s3_deprecated
class USAJobsTest(ImporterTest):
    transformer_class = USAJobsTransformer

    sample_input_document = {
        "PositionURI": "https://www.usajobs.gov:443/GetJob/ViewDetails/333412700",
        "QualificationSummary": "Qualifications: Citizenship: Must be a United States Citizen (non-citizens may be appointed when it is not possible to recruit qualified citizens). English-language proficiency: Must be proficient in spoken and written English. Education: Degree of doctor of medicine or an equivalent degree resulting from a course of education in medicine or osteopathic medicine. The degree must have been obtained from one of the schools approved by the Secretary of Veterans Affairs the year in which the course of study was completed. Approved schools are:\n(1) Schools of medicine holding regular institutional membership in the Association of American Colleges for the year the degree was granted; (2) Schools of osteopathic medicine approved by the American Osteopathic Association for the year which the degree was granted; (3) Schools (including foreign schools accepted by the licensing body of the State, Territory or Commonwealth or the District of Columbia as qualifying for full or unrestricted licensure. Licensure and Registration: Current, full and unrestricted license to practice medicine or surgery in the state, territory or commonwealth of the United States or in the District of Columbia. All individuals: (1) Must meet basic qualification standards for appointment. Clinical privileges must be recommended by the chief, staff/professional standards board as granted by the director of the Philadelphia VA Medical Center. Incumbent must provide complete credentials and privileging information in a timely expeditious way. Boardcertification or board eligibility inInternal medicineis required. Recent critical care or emergency medicine experience, which may include the period of residency training, as well as BCLS and ACLS certification is mandatory. Incumbent must be competent in performing invasive procedures including central venous catheterization, arterial catheterization, thoracentesis, paracentesis, and lumbar puncture. REFERENCE: VA Handook 5005/27, Part II, Appendix G2 - Physician Qualification Standard",
        "PositionRemuneration": [{"RateIntervalCode": "Per Year", "MaximumRange": "240000.0", "MinimumRange": "150000.0"}],
        "JobCategory": [{"Code": "0602", "Name": "Medical Officer"}],
        "PositionID": "PHL-13-803793",
        "PositionLocation": [{"Latitude": 39.95227, "Longitude": -75.16237, "CountrySubDivisionCode": "Pennsylvania", "CityName": "Philadelphia, Pennsylvania", "CountryCode": "United States", "LocationName": "Philadelphia, Pennsylvania"}],
        "PositionOfferingType": [{"Code": "15317", "Name": "Excepted Service Permanent"}],
        "UserArea": {
            "IsRadialSearch": False,
            "Details": {
                "WhoMayApply": {
                    "Code": "15514",
                    "Name": "Non-citizens may be appointed when it is not possible to recruit qualified US citizens"
                },
                "LowGrade": "00",
                "HighGrade": "00",
                "JobSummary": "To fulfill President Lincoln's promise &#150; \"To care for him who shall have borne the battle, and for his widow, and his orphan\" &#150; by serving and honoring the men and women who are America's Veterans.\nHow would you like to become a part of a team providing compassionate care to Veterans?\nThis is an open continuous announcement.We accept applications for this occupation on an ongoing basis; qualified applicants will be considered as vacancies become available. As a VA physician, your opportunities are endless. With many openings in the multiple functions of VA, you will have a wide range of opportunities at your fingertips. Not only is it the largest, most technologically advanced integrated health care system in the nation, but we also provide many other services to Veterans through the Benefits Administration and National Cemeteries. Applicant's education and length of practice (experience) will be considered by aProfessional Standards Board and Compensation Panelin determining the grade and salary of the applicant selected.\nSpecial Employment Consideration: VA encourages persons with disabilities to apply."
            }
        },
        "OrganizationName": "Veterans Affairs, Veterans Health Administration",
        "PositionLocationDisplay": "Philadelphia, Pennsylvania",
        "PublicationStartDate": "2016-07-14",
        "ApplyURI": ["https://www.usajobs.gov:443/GetJob/ViewDetails/333412700?PostingChannelID=RESTAPI"],
        "PositionSchedule": [{"Code": "1", "Name": "Full Time"}],
        "ApplicationCloseDate": "2017-07-13",
        "PositionTitle": "Staff Physician (MICU Hospitalist)",
        "PositionEndDate": "2017-07-13",
        "JobGrade": [{"Code": "VM"}],
        "DepartmentName": "Department Of Veterans Affairs",
        "PositionStartDate": "2016-07-14",
        "PositionFormattedDescription": [{"Label": "Dynamic Teaser", "LabelDescription": "Hit highlighting for keyword searches."}]
    }

    def setUp(self):
        self.connection = boto.connect_s3()
        self.bucket_name = 'usajobs'
        self.prefix = 'akey'
        self.transformer = self.transformer_class(
            bucket_name=self.bucket_name,
            prefix=self.prefix,
            partner_id='US',
            s3_conn=self.connection,
        )

    def test_iterate(self):
        """Test that records from all files are properly returned or excluded
        according to the given date range"""
        bucket = self.connection.create_bucket(self.bucket_name)
        mock_data = {
            '2014Q4': [
                {'PositionStartDate': '2014-12-15', 'PositionEndDate': '2014-12-28'},
                {'PositionStartDate': '2014-11-15', 'PositionEndDate': '2014-12-15'},
            ],
            '2015Q1': [
                {'PositionStartDate': '2015-02-01', 'PositionEndDate': '2015-03-01'},
                {'PositionStartDate': '2015-01-15', 'PositionEndDate': '2015-02-15'},
            ]
        }

        for keyname, rows in mock_data.items():
            for row in rows:
                key = boto.s3.key.Key(
                    bucket=bucket,
                    name='{}/{}/{}.json'.format(self.prefix, keyname, str(uuid.uuid4()))
                )
                stream = StringIO()
                stream.write(json.dumps(row))
                stream.write('\n')
                stream.seek(0)
                key.set_contents_from_file(stream)

        postings = [
            posting
            for posting in self.transformer._iter_postings(quarter='2015Q1')
        ]
        assert len(postings) == 2

    def test_transform(self):
        """Test that the required fields are properly mapped
        from the input data"""
        transformed = self.transformer._transform(self.sample_input_document)
        assert transformed['title'] == 'Staff Physician (MICU Hospitalist)'
        assert transformed['jobLocation']['address']['addressLocality'] == \
            'Philadelphia, Pennsylvania'
        assert transformed['jobLocation']['address']['addressRegion'] == 'Pennsylvania'
        assert transformed['datePosted'] == '2016-07-14'
        assert transformed['validThrough'] == '2017-07-13T00:00:00'
        assert transformed['hiringOrganization'] == {
            '@type': 'Organization',
            'name': 'Veterans Affairs, Veterans Health Administration',
            'department': {
                '@type': 'Organization',
                'name': 'Department Of Veterans Affairs',
            }

        }
        assert transformed['baseSalary'] == {
            '@type': 'MonetaryAmount',
            'minValue': 150000.0,
            'maxValue': 240000.0,
        }
        assert transformed['url'] == "https://www.usajobs.gov:443/GetJob/ViewDetails/333412700"
        assert 'To fulfill President Lincoln' in transformed['description']
