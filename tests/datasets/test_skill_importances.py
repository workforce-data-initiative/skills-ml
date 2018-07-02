import tempfile
import pandas as pd
from mock import patch
import io

from tests import utils

from skills_ml.datasets.skill_importances.onet import OnetSkillImportanceExtractor
from skills_utils.hash import md5
from skills_ml.storage import FSStore


def test_onet_skill_extractor():
    skills_content = [
        ['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Data Value', 'N', 'Standard Error', 'Lower CI Bound', 'Upper CI Bound', 'Recommend Suppress', 'Not Relevant', 'Date', 'Domain Source'],
        ['11-1011.00', '2.A.1.a', 'Reading Comprehension', 'IM', '4.12', '8', '0.13', '3.88', '4.37', 'N', 'n/a', '07/2014', 'Analyst'],
        ['11-1011.00', '2.A.1.a', 'Reading Comprehension', 'LV', '4.75', '8', '0.16', '4.43', '5.07', 'N', 'N', '07/2014', 'Analyst'],
        ['11-1011.00', '2.A.1.b', 'Active Listening', 'IM', '4.12', '8', '0.13', '3.88', '4.37', 'N', 'n/a', '07/2014', 'Analyst'],
        ['11-1011.00', '2.A.1.b', 'Active Listening', 'LV', '-4.88', '8', '0.23', '4.43', '5.32', 'N', 'N', '07/2014', 'Analyst'],
    ]

    abilities_content = [
        ['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Data Value', 'N', 'Standard Error', 'Lower CI Bound', 'Upper CI Bound', 'Recommend Suppress', 'Not Relevant', 'Date', 'Domain Source'],
        ['11-1011.00', '1.A.1.a.1', 'Oral Comprehension', 'IM', '4.50', '8', '0.19', '4.13', '4.87', 'N', 'n/a', '07/2014', 'Analyst'],
        ['11-1011.00', '1.A.1.a.1', 'Oral Comprehension', 'LV', '4.88', '8', '0.13', '4.63', '5.12', 'N', 'Y', '07/2014', 'Analyst'],
        ['11-1011.00', '1.A.1.a.2', 'Written Comprehension', 'IM', '4.25', '8', '0.16', '3.93', '4.57', 'N', 'n/a', '07/2014', 'Analyst'],
        ['11-1011.00', '1.A.1.a.2', 'Written Comprehension', 'LV', '4.62', '8', '0.18', '4.27', '4.98', 'N', 'N', '07/2014', 'Analyst'],
        ['11-2031.00', '1.A.1.a.3', 'Written Comprehension', 'IM', '4.25', '8', '0.16', '3.93', '4.57', 'N', 'n/a', '07/2014', 'Analyst'],
        ['11-2031.00', '1.A.1.a.3', 'Written Comprehension', 'LV', '4.62', '8', '0.18', '4.27', '4.98', 'N', 'N', '07/2014', 'Analyst'],
    ]

    knowledge_content = [
        ['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Data Value', 'N', 'Standard Error', 'Lower CI Bound', 'Upper CI Bound', 'Recommend Suppress', 'Not Relevant', 'Date', 'Domain Source'],
        ['11-1011.00', '2.C.1.a', 'Administration and Management', 'IM', '4.75', '27', '0.09', '4.56', '4.94', 'N', 'n/a', '07/2014', 'Incumbent'],
        ['11-1011.00', '2.C.1.a', 'Administration and Management', 'LV', '6.23', '27', '0.17', '5.88', '6.57', 'N', 'N', '07/2014', 'Incumbent'],
        ['11-1011.00', '2.C.1.b', 'Clerical', 'IM', '2.66', '27', '0.22', '2.21', '3.11', 'N', 'n/a', '07/2014', 'Incumbent'],
        ['11-1011.00', '2.C.1.b', 'Clerical', 'LV', '3.50', '27', '0.41', '2.66', '4.34', 'N', 'N', '07/2014', 'Incumbent'],
    ]

    tools_content = [
        ['O*NET-SOC Code', 'T2 Type', 'T2 Example', 'Commodity Code', 'Commodity Title'],
        ['11-1011.00', 'Tools', '10-key calculators', '44101809', 'Desktop calculator'],
        ['11-1011.00', 'Tools', 'Desktop computers', '43211507', 'Desktop computers'],
        ['11-1011.00', 'Tools', 'Laptop computers', '43211503', 'Notebook computers'],
        ['11-1011.00', 'Tools', 'Personal computers', '43211508', 'Personal computers'],
        ['11-1011.00', 'Tools', 'Personal digital assistants PDA', '43211504', 'Personal digital assistant PDAs or organizers'],
        ['11-1011.00', 'Tools', 'Smartphones', '43191501', 'Mobile phones'],
        ['11-1011.00', 'Tools', 'Universal serial bus USB flash drives', '43201813', 'High capacity removable media drives'],
        ['11-1011.00', 'Technology', 'Adobe Systems Adobe Acrobat software', '43232202', 'Document management software'],
        ['11-1011.00', 'Technology', 'AdSense Tracker', '43232306', 'Data base user interface and query software'],
        ['11-1011.00', 'Technology', 'Blackbaud The Raiser\'s Edge', '43232303', 'Customer relationship management CRM software'],
    ]



    class MockOnetDownloader(object):
        def download(self, source_file):
            fake_data_lookup = {
                'Skills': skills_content,
                'Abilities': abilities_content,
                'Knowledge': knowledge_content,
                'Tools and Technology': tools_content,
            }
             
            with utils.makeNamedTemporaryCSV(
                fake_data_lookup[source_file],
                '\t'
            ) as tempname:
                with open(tempname) as fh:
                    return fh.read()

    with patch('skills_ml.datasets.skill_importances.onet.OnetToMemoryDownloader', MockOnetDownloader):
        with tempfile.TemporaryDirectory() as output_dir:
            storage = FSStore(output_dir)
            extractor = OnetSkillImportanceExtractor(
                output_dataset_name='skills',
                storage=storage,
                hash_function=md5
            )
            extractor.run()
            pdin = io.StringIO(storage.load('skills.tsv').decode('utf-8'))
            output = pd.read_csv(pdin, sep='\t').T.to_dict().values()

            # +24 base rows in input across the K,S,A,T files
            assert len(output) == 24

            # make sure uuid is hashed version of the KSA
            for row in output:
                assert row['nlp_a'] == md5(row['ONET KSA'])
                # otherwise, this is a simple concat so not much to assert
                # we do use these rows though so make sure they're there
                assert 'O*NET-SOC Code' in row
                assert 'ONET KSA' in row
