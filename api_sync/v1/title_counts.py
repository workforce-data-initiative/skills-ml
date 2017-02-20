from sqlalchemy.orm import sessionmaker
import csv
import hashlib
import logging

from .models import TitleCount, Quarter


def load_title_counts(filename, year, quarter, db_engine):
    session = sessionmaker(db_engine)()

    quarter_record = session.query(Quarter)\
        .filter_by(year=year, quarter=quarter)\
        .first()
    if not quarter_record:
        quarter_record = Quarter(year=year, quarter=quarter)
        session.add(quarter_record)
        session.commit()

    quarter_id = quarter_record.quarter_id
    session.query(TitleCount).filter_by(quarter_id=quarter_id).delete()

    with open(filename) as f:
        reader = csv.reader(f)
        records = []
        for row in reader:
            if len(row) < 2:
                logging.warning("Skipping %s due to not enough data", row)
                continue
            else:
                title, count = row

            if title == '':
                logging.warning("Skipping %s due to invalid data", row)
                continue

            job_uuid = str(hashlib.md5(title.encode('utf-8')).hexdigest())
            records.append(TitleCount(
                job_uuid=job_uuid,
                job_title=title,
                quarter_id=quarter_id,
                count=count
            ))
        session.bulk_save_objects(records)
        session.commit()
