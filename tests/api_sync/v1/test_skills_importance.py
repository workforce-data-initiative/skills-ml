import testing.postgresql
from sqlalchemy.orm import sessionmaker
from tests import utils

from sqlalchemy import create_engine
from api_sync.v1.models import ensure_db, JobMaster, SkillMaster
from api_sync.v1.skills_importance import load_skills_importance


sample_ksas = [
    [
        '',
        'O*NET-SOC Code',
        'Element ID',
        'ONET KSA',
        'Scale ID',
        'Data Value',
        'N',
        'Standard Error',
        'Lower CI Bound',
        'Upper CI Bound',
        'Recommend Suppress',
        'Not Relevant',
        'Date',
        'Domain Source',
        'skill_uuid'
    ],
    [
        '0',
        '11-1011.00',
        '2.A.1.a',
        'reading comprehension',
        'im',
        '4.12',
        '8',
        '0.13',
        '3.88',
        '4.37',
        'N',
        'n/a',
        '07/2014',
        'Analyst',
        'skill_uuid1'
    ],
    [
        '1',
        '11-1011.00',
        '2.A.1.a',
        'reading comprehension',
        'lv',
        '4.75',
        '8',
        '0.16',
        '4.43',
        '5.07',
        'N',
        'N',
        '07/2014',
        'Analyst',
        'skill_uuid1'
    ],
    [
        '2',
        '11-1011.00',
        '2.A.1.b',
        'active listening',
        'im',
        '4.12',
        '8',
        '0.13',
        '3.88',
        '4.37',
        'N',
        'n/a',
        '07/2014',
        'Analyst',
        'skill_uuid2'
    ],
    [
        '3',
        '11-1011.00',
        '2.A.1.b',
        'active listening',
        'lv',
        '4.88',
        '8',
        '0.23',
        '4.43',
        '5.32',
        'N',
        'N',
        '07/2014',
        'Analyst',
        'skill_uuid2'
    ],
    [
        '3',
        '11-1012.00',
        '2.A.1.b',
        'active listening',
        'lv',
        '4.88',
        '8',
        '0.23',
        '4.43',
        '5.32',
        'N',
        'N',
        '07/2014',
        'Analyst',
        'skill_uuid2'
    ],
]


def test_skills_importance():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        session = sessionmaker(engine)()
        # this task depends on jobs and skills master being loaded
        # so add the needed rows
        session.add(JobMaster('job_uuid', '11-1011.00', '', '', '', ''))
        session.add(SkillMaster(uuid='skill_uuid1'))
        session.add(SkillMaster(uuid='skill_uuid2'))
        session.commit()
        with utils.makeNamedTemporaryCSV(sample_ksas, separator='\t') as fname:
            load_skills_importance(fname, engine)
            # the four rows will be smashed into two which have both LV and IM
            results = [
                row for row in
                engine.execute('select * from skills_importance')
            ]
            assert len(results) == 2
            assert results == [
                ('job_uuid', 'skill_uuid1', 4.75, 4.12),
                ('job_uuid', 'skill_uuid2', 4.88, 4.12)
            ]
