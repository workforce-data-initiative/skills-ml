import csv
from sqlalchemy.orm import sessionmaker

from .models import JobMaster


def load_jobs_master(filename, db_engine):
    with open(filename) as f:
        reader = csv.DictReader(f, delimiter='\t')
        session = sessionmaker(db_engine)()
        for row in reader:
            if row['Original Title'] != row['Title']:
                continue
            job = JobMaster(
                uuid=row['job_uuid'],
                onet_soc_code=row['O*NET-SOC Code'],
                title=row['Title'],
                original_title=row['Original Title'],
                description=row['Description'],
                nlp_a=row['nlp_a']
            )
            session.merge(job)
        session.commit()
