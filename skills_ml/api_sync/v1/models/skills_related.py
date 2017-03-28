# -*- coding: utf-8 -*-

"""Skills Related ORM"""
from . import Base
import sqlalchemy as db

from sqlalchemy.dialects.postgresql import JSONB


class SkillRelated(Base):
    __tablename__ = 'skills_related'

    uuid = db.Column(db.String, primary_key=True)
    related_skills = db.Column(JSONB)

    def __init__(self, uuid, related_skills):
        self.uuid = uuid
        self.related_skills = related_skills

    def __repr__(self):
        return '<uuid {}>'.format(self.uuid)
