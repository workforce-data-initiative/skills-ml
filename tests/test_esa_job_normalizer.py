from algorithms.job_normalizers.esa_jobtitle_normalizer import ESANormalizer
from mock import patch
from tests import utils

occupation_content = [
    ['O*NET-SOC Code', 'Title', 'Description'],
    ['11-1011.00', 'Chief Executives', 'Determine and formulate policies and provide overall direction of companies or private and public sector organizations within guidelines set up by a board of directors or similar governing body. Plan, direct, or coordinate operational activities at the highest level of management with the help of subordinate executives and staff managers.'],
    ['11-1011.03', 'Chief Sustainability Officers', 'Communicate and coordinate with management, shareholders, customers, and employees to address sustainability issues. Enact or oversee a corporate sustainability strategy.'],
    ['11-1021.00', 'General and Operations Managers', 'Plan, direct, or coordinate the operations of public or private sector organizations. Duties and responsibilities include formulating policies, managing daily operations, and planning the use of materials and human resources, but are too diverse and general in nature to be classified in any one functional area of management or administration, such as personnel, purchasing, or administrative services.'],
    ['11-1031.00', 'Legislators', 'Develop, introduce or enact laws and statutes at the local, tribal, State, or Federal level. Includes only workers in elected positions.'],
]


class MockOnetSourceDownloader(object):
    def download(self, *args):
        with utils.makeNamedTemporaryCSV(occupation_content, '\t') as temp:
            return temp


@patch('algorithms.job_normalizers.esa_jobtitle_normalizer.ONET_VERSIONS', ['db_21_0_text'])
def test_esa_normalizer():
    normalizer = ESANormalizer(MockOnetSourceDownloader)
    assert normalizer.normalize_job_title('Staffing Expert')[0]['title'] == 'chief executives'
