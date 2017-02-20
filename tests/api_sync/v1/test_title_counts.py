import testing.postgresql
from sqlalchemy.orm import sessionmaker
from tests import utils

from sqlalchemy import create_engine
from api_sync.v1.models import ensure_db, TitleCount, Quarter
from api_sync.v1.title_counts import load_title_counts

sample_counts = [
    ['customer service representative', 2],
    ['3 paper machine superintendent', 1],
    ['mental health technician fulltime', 1],
    ['customer service marketing reps needed  work at home', 1],
    ['1000 week to start  full benefits 401k', 1],
    ['inside sales representative', 2],
    ['restaurant management opportunities', 2],
    ['construction manager  training available', 3],
]


def test_load_title_counts():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        with utils.makeNamedTemporaryCSV(sample_counts) as fname:
            year = 2014
            quarter = 2
            load_title_counts(fname, year, quarter, engine)

        session = sessionmaker(engine)()

        # make sure the correct amount of rows were created
        assert session.query(TitleCount).count() == 8
        assert session.query(Quarter).count() == 1

        # try a different quarter
        with utils.makeNamedTemporaryCSV(sample_counts) as fname:
            year = 2014
            quarter = 3
            load_title_counts(fname, year, quarter, engine)

        assert session.query(TitleCount).count() == 16
        assert session.query(Quarter).count() == 2

        # overwrite data for a quarter with existing data
        with utils.makeNamedTemporaryCSV(sample_counts) as fname:
            year = 2014
            quarter = 3
            load_title_counts(fname, year, quarter, engine)

        assert session.query(TitleCount).count() == 16
        assert session.query(Quarter).count() == 2
