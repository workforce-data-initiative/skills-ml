from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

from .geographies import Geography
from .geo_title_count import GeoTitleCount
from .jobs_alternate_titles import JobAlternateTitle
from .jobs_importance import JobImportance
from .jobs_master import JobMaster
from .jobs_skills import JobSkill
from .jobs_unusual_titles import JobUnusualTitle
from .quarters import Quarter
from .skills_importance import SkillImportance
from .skills_master import SkillMaster
from .skills_related import SkillRelated
from .title_count import TitleCount

def ensure_db(engine):
    Base.metadata.create_all(engine)
