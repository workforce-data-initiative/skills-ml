# -*- coding: utf-8 -*-

"""Skills Master ORM"""

from . import Base
import sqlalchemy as db


class SkillMaster(Base):
    __tablename__ = 'skills_master'

    uuid = db.Column(db.String, primary_key=True)
    skill_name = db.Column(db.String)
    ksa_type = db.Column(db.String)
    onet_element_id = db.Column(db.String)
    description = db.Column(db.String)
    nlp_a = db.Column(db.String)

    def __repr__(self):
        return '<uuid {}>'.format(self.uuid)
