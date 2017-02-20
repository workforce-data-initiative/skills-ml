import csv
import logging
from sqlalchemy.orm import sessionmaker

from .models import JobMaster, SkillMaster, SkillImportance


def load_skills_importance(filename, db_engine):
    with open(filename) as f:
        reader = csv.DictReader(f, delimiter='\t')
        session = sessionmaker(db_engine)()
        found_importance = False
        found_level = False
        level = ''
        importance = ''

        for row in reader:
            if not found_importance:
                job = session.query(JobMaster)\
                    .filter_by(onet_soc_code=row['O*NET-SOC Code'])\
                    .first()
                if not job:
                    logging.warning('Job %s not found, skipping', row)
                    found_level = False
                    found_importance = False
                    level = ''
                    importance = ''
                    continue

                if row['Scale ID'] == 'im':
                    importance = row['Data Value']
                    found_importance = True
            elif not found_level and found_importance:
                if row['Scale ID'] == 'lv':
                    level = row['Data Value']
                    found_level = True
                    if not session.query(SkillMaster).get(row['skill_uuid']):
                        logging.warning('Skill %s not found, skipping', row)
                        found_level = False
                        found_importance = False
                        level = ''
                        importance = ''
                        continue

                    skills_importance = SkillImportance(
                        job.uuid,
                        row['skill_uuid'],
                        level,
                        importance
                    )
                    session.merge(skills_importance)

                    # reset importance and level
                    found_level = False
                    found_importance = False
                    level = ''
                    importance = ''
        session.commit()
