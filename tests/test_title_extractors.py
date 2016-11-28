import contextlib
import tempfile
import pandas as pd

from tests import utils

from algorithms.title_extractors.onet import OnetTitleExtractor
from utils.hash import md5


def test_onet_title_extractor():
    occupation_content = [
        ['O*NET-SOC Code', 'Title', 'Description'],
        ['11-1011.00', 'Chief Executives', 'Not important'],
        ['11-1011.03', 'Chief Sustainability Officers', 'Not important'],
        ['11-1021.00', 'General and Operations Managers', 'Not important'],
        ['11-1031.00', 'Legislators', 'Not important'],
    ]

    alternate_title_content = [
        ['O*NET-SOC Code', 'Alternate Title', 'Short Title', 'Source(s)'],
        ['11-1011.00', 'Aeronautics Commission Director', 'n/a', '08'],
        ['11-1011.00', 'Agricultural Services Director', 'n/a', '08'],
        ['11-1011.00', 'Alcohol and Drug Abuse Assistance Admin', 'n/a', '08'],
    ]

    sample_content = [
        ['O*NET-SOC Code', 'Reported Job Title', 'Shown in My Next Move'],
        ['11-1011.00', 'Chief Diversity Officer (CDO)', 'N'],
        ['11-1011.00', 'Chief Executive Officer (CEO)', 'Y'],
        ['11-1011.00', 'Chief Financial Officer (CFO)', 'Y'],
    ]

    class MockOnetTitleCache(object):
        @contextlib.contextmanager
        def ensure_file(self, dataset):
            fake_data_lookup = {
                'Sample of Reported Titles.txt': sample_content,
                'Occupation Data.txt': occupation_content,
                'Alternate Titles.txt': alternate_title_content,
            }
            with utils.makeNamedTemporaryCSV(
                fake_data_lookup[dataset],
                '\t'
            ) as temp:
                yield temp

    with tempfile.NamedTemporaryFile(mode='w+') as outputfile:
        extractor = OnetTitleExtractor(
            output_filename=outputfile.name,
            onet_source=MockOnetTitleCache(),
            hash_function=md5
        )
        extractor.run()
        outputfile.seek(0)
        output = pd.read_csv(outputfile, sep='\t').T.to_dict().values()

        # the new file should be the three files concatenated
        assert len(output) == 10

        # for non-occupations, original title should be occupation
        assert next(
            row['Original Title']
            for row in output
            if row['Title'] == 'Aeronautics Commission Director'
        ) == 'Chief Executives'

        # for occupations, the original titles should also be occupation
        assert next(
            row['Original Title']
            for row in output
            if row['Title'] == 'Chief Executives'
        ) == 'Chief Executives'

        # make sure uuid is hashed version of the title
        for row in output:
            assert row['job_uuid'] == md5(row['Original Title'])

        # make sure nlp_a is cleaned version of title
        assert next(
            row['nlp_a']
            for row in output
            if row['Title'] == 'Chief Diversity Officer (CDO)'
        ) == 'chief diversity officer cdo'
