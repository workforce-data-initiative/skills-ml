import testing.postgresql
from tests import utils

from sqlalchemy import create_engine
from api_sync.v1.models import ensure_db
from api_sync.v1.skills_master import load_skills_master

sample_input = [
    ['', 'O*NET-SOC Code', 'Element ID', 'ONET KSA', 'ksa_type', 'Description', 'skill_uuid', 'nlp_a'],
    ['0', '11-1011.00', '2.a.1.a', 'reading comprehension', 'skill', 'understanding written sentences and paragraphs in work related documents.', '2c77c703bd66e104c78b1392c3203362', 'reading comprehension'],
    ['1', '11-1011.00', '2.a.1.b', 'active listening', 'skill', 'giving full attention to what other people are saying, taking time to understand the points being made, asking questions as appropriate, and not interrupting at inappropriate times.', 'a636cb69257dcec699bce4f023a05126', 'active listening'],
    ['2', '11-1011.00', '2.a.1.c', 'writing', 'skill', 'communicating effectively in writing as appropriate for the needs of the audience.', '1cea5345d284f36245a94301b114b27c', 'writing'],
    ['3', '11-1011.00', '2.a.1.d', 'speaking', 'skill', 'talking to others to convey information effectively.', 'd1715efc5a67ac1c988152b8136e3dfa', 'speaking'],
    ['4', '11-1011.00', '2.a.1.e', 'mathematics', 'skill', 'using mathematics to solve problems.', '6ae28a55456b101be8261e5dee44cd3e', 'mathematics'],
    ['5', '11-1011.00', '2.a.1.f', 'science', 'skill', 'using scientific rules and methods to solve problems.', 'fb5c7f9bb4b32ce2f3bff4662f1ab27b', 'science'],
    ['6', '11-1011.00', '2.a.2.a', 'critical thinking', 'skill', 'using logic and reasoning to identify the strengths and weaknesses of alternative solutions,  conclusions or approaches to problems.', '20784bf09c9fe614603ad635e6093ede', 'critical thinking'],
    ['7', '11-1011.00', '2.a.2.b', 'active learning', 'ability', 'understanding the implications of new information for both current and future problem-solving and decision-making.', 'e98ada9866a495536ba6348ccec73915', 'active learning'],
]

new_input = [
    ['', 'O*NET-SOC Code', 'Element ID', 'ONET KSA', 'ksa_type', 'Description', 'skill_uuid', 'nlp_a'],
    ['0', '11-1011.00', '2.a.1.a', 'reading comprehension', 'skill', 'an updated description', '2c77c703bd66e104c78b1392c3203362', 'reading comprehension'],

]


def test_load_skills_master():
    def num_rows():
        return len([row for row in engine.execute('select * from skills_master')])

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        with utils.makeNamedTemporaryCSV(sample_input, separator='\t') as fname:
            load_skills_master(fname, engine)
            assert num_rows() == 8

        # test that a new run with updated data updates the matched row
        with utils.makeNamedTemporaryCSV(new_input, separator='\t') as fname:
            load_skills_master(fname, engine)
            assert num_rows() == 8
            updated_skill_desc_query = '''
select description
from skills_master
where uuid = '2c77c703bd66e104c78b1392c3203362'
'''
            assert [
                row[0] for row
                in engine.execute(updated_skill_desc_query)
            ][0] == 'an updated description'
