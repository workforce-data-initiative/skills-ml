import contextlib
import tempfile
import pandas as pd

from tests import utils

from algorithms.skill_extractors.onet_ksas import OnetSkillExtractor
from utils.hash import md5


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
    ]

    cmr_content = [
        ['Element ID', 'Element Name', 'Description'],
        ['1', 'Worker Characteristics', 'Worker Characteristics'],
        ['1.A', 'Abilities', 'Enduring attributes of the individual that influence performance'],
        ['1.A.1', 'Cognitive Abilities', 'Abilities that influence the acquisition and application of knowledge in problem solving'],
        ['1.A.1.a', 'Verbal Abilities', 'Abilities that influence the acquisition and application of verbal information in problem solving'],
        ['1.A.1.a.1', 'Oral Comprehension', 'The ability to listen to and understand information and ideas presented through spoken words and sentences.'],
        ['1.A.1.a.2', 'Written Comprehension', 'The ability to read and understand information and ideas presented in writing.'],
        ['1.A.1.a.3', 'Oral Expression', 'The ability to communicate information and ideas in speaking so others will understand.'],
        ['1.A.1.a.4', 'Written Expression', 'The ability to communicate information and ideas in writing so others will understand.'],
    ]

    class MockOnetSkillCache(object):
        @contextlib.contextmanager
        def ensure_file(self, dataset):
            fake_data_lookup = {
                'Skills.txt': skills_content,
                'Abilities.txt': abilities_content,
                'Knowledge.txt': knowledge_content,
                'Tools and Technology.txt': tools_content,
                'Content Model Reference.txt': cmr_content
            }
            with utils.makeNamedTemporaryCSV(
                fake_data_lookup[dataset],
                '\t'
            ) as temp:
                yield temp

    with tempfile.NamedTemporaryFile(mode='w+') as outputfile:
        extractor = OnetSkillExtractor(
            output_filename=outputfile.name,
            onet_source=MockOnetSkillCache(),
            hash_function=md5
        )
        extractor.run()
        outputfile.seek(0)
        output = pd.read_csv(outputfile, sep='\t').T.to_dict().values()

        # +17 base rows in input across the K,S,A,T files
        # -7 rows that don't have scale LV
        # -1 row that has 'Not Relevant' marked Y
        # -1 row that has 'Data Value' below 0
        # -1 row that is a dupe
        assert len(output) == 7

        assert len([row for row in output if row['ksa_type'] == 'knowledge']) == 2
        assert len([row for row in output if row['ksa_type'] == 'skill']) == 1
        assert len([row for row in output if row['ksa_type'] == 'ability']) == 1
        assert len([row for row in output if row['ksa_type'] == 'tool']) == 3

        # make sure uuid is hashed version of the KSA
        for row in output:
            assert row['skill_uuid'] == md5(row['ONET KSA'])

        # make sure nlp_a is cleaned version of skill
        assert next(
            row['nlp_a']
            for row in output
            if row['ONET KSA'] == '10-key calculators'
        ) == '10key calculators'

        # make sure duplicate entries pick first SOC Code
        assert next(
            row['O*NET-SOC Code']
            for row in output
            if row['ONET KSA'] == 'written comprehension'
        ) == '11-1011.00'

        # make sure duplicate entries pick first element id
        assert next(
            row['Element ID']
            for row in output
            if row['ONET KSA'] == 'written comprehension'
        ) == '1.a.1.a.2'
