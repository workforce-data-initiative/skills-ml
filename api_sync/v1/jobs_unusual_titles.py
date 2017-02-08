from sqlalchemy.orm import sessionmaker
import hashlib
import logging
import csv
from .models import JobMaster, JobUnusualTitle


def load_jobs_unusual_titles(fname, db_engine):
    session = sessionmaker(db_engine)()
    with open(fname) as f:
        reader = csv.reader(f, delimiter='\t')
        for title, description, soc_code in reader:
            title_uuid = str(hashlib.md5(title.encode('utf-8')).hexdigest())
            job = session.query(JobMaster)\
                .filter_by(onet_soc_code=soc_code)\
                .first()
            if job is not None:
                unusual_job_title = JobUnusualTitle(
                    title_uuid,
                    title,
                    description,
                    job.uuid
                )

                try:
                    session.merge(unusual_job_title)
                    session.commit()
                except:
                    logging.warning(
                        'Could not add unusual job title %s',
                        title
                    )
            else:
                logging.warning(
                    'Could not find job for unusual title %s, code %s',
                    title,
                    soc_code
                )
