# -*- coding: utf-8 -*-

from . import Base
import sqlalchemy as db


class JobAlternateTitle(Base):
    __tablename__ = 'jobs_alternate_titles'

    uuid = db.Column(db.String, primary_key=True)
    title = db.Column(db.String)
    nlp_a = db.Column(db.String)
    job_uuid = db.Column(db.String, db.ForeignKey('jobs_master.uuid'))

    def __init__(self, uuid, title, nlp_a, job_uuid):
        self.uuid = uuid
        self.title = title
        self.nlp_a = nlp_a
        self.job_uuid = job_uuid

    def __repr__(self):
        return '<uuid {}>'.format(self.uuid)
