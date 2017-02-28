from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import os
import yaml


def get_apiv1_dbengine():
    return create_engine(get_apiv1_dburl())


def get_apiv1_dburl():
    dburl = os.environ.get('API_V1_DB_URL', None)
    if not dburl:
        config_filename = os.path.join(
            os.path.dirname(__file__),
            '../',
            'api_v1_db_config.yaml'
        )

        with open(config_filename) as f:
            config = yaml.load(f)
            dbconfig = {
                'host': config['PGHOST'],
                'username': config['PGUSER'],
                'database': config['PGDATABASE'],
                'password': config['PGPASSWORD'],
                'port': config['PGPORT'],
            }
            dburl = URL('postgres', **dbconfig)
    return dburl
