import csv
from sqlalchemy.orm import sessionmaker

from .models import SkillMaster


def load_skills_master(filename, db_engine):
    with open(filename) as f:
        reader = csv.DictReader(f, delimiter='\t')
        session = sessionmaker(db_engine)()
        for row in reader:
            session.merge(SkillMaster(
                uuid=row['skill_uuid'],
                skill_name=row['ONET KSA'],
                ksa_type=row['ksa_type'],
                onet_element_id=row['Element ID'],
                description=row['Description'],
                nlp_a=row['nlp_a']
            ))
        session.commit()
