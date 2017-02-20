import testing.postgresql
from airflow import configuration
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from airflow import models
from mock import patch
from dags.api_sync_v1 import dag
from api_sync.v1.models import JobMaster,\
    SkillMaster,\
    SkillImportance,\
    GeoTitleCount,\
    TitleCount
import os
import logging

DEFAULT_DATE = datetime(2013, 5, 1)
configuration.test_mode()


def test_dag():
    with testing.postgresql.Postgresql() as postgresql:
        configuration.test_mode()

        with patch.dict(os.environ, {
            'API_V1_DB_URL': postgresql.url(),
            'OUTPUT_FOLDER': 'tests/api_sync/v1/input'
        }):
            bag = models.DagBag()
            bag.bag_dag(dag, dag, dag)
            dag.clear(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)
            dag.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, local=True)

            engine = create_engine(postgresql.url())
            session = sessionmaker(engine)()
            logging.warning(session.query(JobMaster).all())
            num_jobs = session.query(JobMaster).count()
            assert num_jobs > 1
            num_skills = session.query(SkillMaster).count()
            assert num_skills > 1
            num_importances = session.query(SkillImportance).count()
            assert num_importances > 1
            assert session.query(GeoTitleCount).count() > 1
            assert session.query(TitleCount).count() > 1

            # make sure non-temporal data doesn't load twice for a different quarter
            new_date = datetime(2014, 5, 1)
            dag.clear(start_date=new_date, end_date=new_date)
            dag.run(start_date=new_date, end_date=new_date, local=True)
            assert session.query(JobMaster).count() == num_jobs
            assert session.query(SkillMaster).count() == num_skills
            assert session.query(SkillImportance).count() == num_importances
