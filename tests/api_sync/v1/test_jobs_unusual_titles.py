import testing.postgresql
from sqlalchemy.orm import sessionmaker
from tests.utils import makeNamedTemporaryCSV

from sqlalchemy import create_engine
from api_sync.v1.models import ensure_db, JobMaster, JobUnusualTitle
from api_sync.v1.jobs_unusual_titles import load_jobs_unusual_titles


sample_titles = [
    ['digital overlord', 'website manager', '15-1199.03'],
    ['retail jedi', 'shop assistant', '41-2031.00']
]


def test_load_jobs_unusual_titles():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        session = sessionmaker(engine)()
        with makeNamedTemporaryCSV(sample_titles, separator='\t') as fname:
            session.add(JobMaster('abcd', '41-2031.00', '', '', '', ''))
            session.commit()
            load_jobs_unusual_titles(fname, engine)
        assert session.query(JobUnusualTitle).count() == 1
        assert session.query(JobUnusualTitle)\
            .filter_by(job_uuid='abcd')\
            .count() == 1
