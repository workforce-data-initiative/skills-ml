import testing.postgresql
from tests import utils

from sqlalchemy import create_engine
from api_sync.v1.models import ensure_db
from api_sync.v1.jobs_master import load_jobs_master


sample_input = [
    ['', 'O*NET-SOC Code', 'Title', 'Original Title', 'Description', 'job_uuid', 'nlp_a'],
    ['0', '11-1011.00', 'Chief Executives', 'Chief Executives', 'Determine and formulate policies and provide overall direction of companies or private and public sector organizations within guidelines set up by a board of directors or similar governing body. Plan, direct, or coordinate operational activities at the highest level of management with the help of subordinate executives and staff managers.', 'e4063de16cae5cf29207ca572e3a891d', 'chief executives'],
    ['1', '11-1011.03', 'Chief Sustainability Officers', 'Chief Sustainability Officers', 'Communicate and coordinate with management, shareholders, customers, and employees to address sustainability issues. Enact or oversee a corporate sustainability strategy.', 'b4155ade06cff632fb89ff03057b3107', 'chief sustainability officers'],
    ['2', '11-1021.00', 'General and Operations Managers', 'General and Operations Managers', 'Plan, direct, or coordinate the operations of public or private sector organizations. Duties and responsibilities include formulating policies, managing daily operations, and planning the use of materials and human resources, but are too diverse and general in nature to be classified in any one functional area of management or administration, such as personnel, purchasing, or administrative services.', '7470171b25f50df9361eb1a0afab6ff4', 'general and operations managers'],
    ['1110', '11-1011.00', 'Aeronautics Commission Director', 'Chief Executives', 'Determine and formulate policies and provide overall direction of companies or private and public sector organizations within guidelines set up by a board of directors or similar governing body. Plan, direct, or coordinate operational activities at the highest level of management with the help of subordinate executives and staff managers.', 'e4063de16cae5cf29207ca572e3a891d', 'aeronautics commission director'],
]


def test_load_jobs_master():
    def num_rows():
        return len([row for row in engine.execute('select * from jobs_master')])

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        with utils.makeNamedTemporaryCSV(sample_input, separator='\t') as fname:
            load_jobs_master(fname, engine)
            assert num_rows() == 3
            # make sure the occupation (first version) of Chief Executives is used
            title_query = '''
select title
from jobs_master
where uuid = 'e4063de16cae5cf29207ca572e3a891d'
'''
            assert [
                row[0] for row
                in engine.execute(title_query)
            ][0] == 'Chief Executives'
