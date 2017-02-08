import testing.postgresql
from sqlalchemy.orm import sessionmaker
from tests import utils

from sqlalchemy import create_engine
from api_sync.v1.models import ensure_db, GeoTitleCount, Geography, Quarter
from api_sync.v1.geo_title_counts import load_geo_title_counts

sample_counts = [
    [41700, 'customer service representative', 2],
    [38060, '3 paper machine superintendent', 1],
    [38940, 'mental health technician fulltime', 1],
    [23580, 'customer service marketing reps needed  work at home', 1],
    [44220, '1000 week to start  full benefits 401k', 1],
    [38060, 'inside sales representative', 2],
    [26420, 'restaurant management opportunities', 2],
    [37100, 'construction manager  training available', 3],
]


def test_load_geo_title_counts():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        with utils.makeNamedTemporaryCSV(sample_counts) as fname:
            year = 2014
            quarter = 2
            load_geo_title_counts(fname, year, quarter, engine)

        session = sessionmaker(engine)()

        # make sure the correct amount of rows were created
        assert session.query(GeoTitleCount).count() == 8
        assert session.query(Geography).count() == 7
        assert session.query(Quarter).count() == 1

        # try a different quarter
        with utils.makeNamedTemporaryCSV(sample_counts) as fname:
            year = 2014
            quarter = 3
            load_geo_title_counts(fname, year, quarter, engine)

        assert session.query(GeoTitleCount).count() == 16
        assert session.query(Geography).count() == 7
        assert session.query(Quarter).count() == 2

        # overwrite data for a quarter with existing data
        with utils.makeNamedTemporaryCSV(sample_counts) as fname:
            year = 2014
            quarter = 3
            load_geo_title_counts(fname, year, quarter, engine)

        assert session.query(GeoTitleCount).count() == 16
        assert session.query(Geography).count() == 7
        assert session.query(Quarter).count() == 2
