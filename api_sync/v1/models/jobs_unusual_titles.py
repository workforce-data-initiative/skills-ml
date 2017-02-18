# -*- coding: utf-8 -*-

"""Jobs Unusual Titles ORM"""

import sqlalchemy as db
from . import Base


class JobUnusualTitle(Base):
    __tablename__ = 'jobs_unusual_titles'

    uuid = db.Column(db.String, primary_key=True)
    title = db.Column(db.String)
    description = db.Column(db.String)
    job_uuid = db.Column(db.String, db.ForeignKey('jobs_master.uuid'))

    def __init__(self, uuid, title, description, job_uuid):
        self.uuid = uuid
        self.title = title
        self.description = description
        self.job_uuid = job_uuid

    def __repr__(self):
        return '<uuid {}>'.format(self.uuid)
