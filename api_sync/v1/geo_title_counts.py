from sqlalchemy.orm import sessionmaker
import csv
import hashlib
import logging

from .models import GeoTitleCount, Geography, Quarter


def load_geo_title_counts(filename, year, quarter, db_engine):
    session = sessionmaker(db_engine)()
    quarter_record = session.query(Quarter)\
        .filter_by(year=year, quarter=quarter)\
        .first()
    if not quarter_record:
        quarter_record = Quarter(year=year, quarter=quarter)
        session.add(quarter_record)
        session.commit()

    quarter_id = quarter_record.quarter_id

    session.query(GeoTitleCount).filter_by(quarter_id=quarter_id).delete()

    with open(filename) as f:
        reader = csv.reader(f)
        records = []
        for row in reader:
            if len(row) < 3:
                logging.warning("Skipping %s due to not enough data", row)
                continue
            else:
                cbsa, title, count = row

            if cbsa == '' or title == '':
                logging.warning("Skipping %s due to invalid data", row)
                continue

            kwargs = {
                'geography_name': cbsa,
                'geography_type': 'CBSA'
            }
            geography = session.query(Geography).filter_by(**kwargs).first()
            if not geography:
                geography = Geography(**kwargs)
                session.add(geography)
                session.commit()

            job_uuid = str(hashlib.md5(title.encode('utf-8')).hexdigest())
            records.append(GeoTitleCount(
                job_uuid=job_uuid,
                job_title=title,
                quarter_id=quarter_id,
                geography_id=geography.geography_id,
                count=count
            ))
        session.bulk_save_objects(records)
        session.commit()
