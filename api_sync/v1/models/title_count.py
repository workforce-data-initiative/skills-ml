# -*- coding: utf-8 -*-

from . import Base
import sqlalchemy as db


class TitleCount(Base):
    __tablename__ = 'title_counts'

    job_uuid = db.Column(db.String, primary_key=True)
    quarter_id = db.Column(db.SmallInteger, db.ForeignKey('quarters.quarter_id'), primary_key=True)
    job_title = db.Column(db.String)
    count = db.Column(db.Integer)

    def __repr__(self):
        return '<TitleCount {}/{}>'.format(
            self.quarter_id, self.job_uuid, self.count
        )
