# -*- coding: utf-8 -*-

"""Jobs Master ORM"""

from . import Base
import sqlalchemy as db
from sqlalchemy.orm import relationship


class JobMaster(Base):
    __tablename__ = 'jobs_master'

    uuid = db.Column(db.String, primary_key=True)
    onet_soc_code = db.Column(db.String)
    title = db.Column(db.String)
    original_title = db.Column(db.String)
    description = db.Column(db.String)
    nlp_a = db.Column(db.String)
    alternate_titles = relationship('JobAlternateTitle', backref='job', lazy='dynamic')
    unusual_titles = relationship('JobUnusualTitle', backref='job', lazy='dynamic')

    def __init__(self, uuid, onet_soc_code, title, original_title, description, nlp_a):
        self.uuid = uuid
        self.onet_soc_code = onet_soc_code
        self.title = title
        self.original_title = original_title
        self.description = description
        self.nlp_a = nlp_a

    def __repr__(self):
        return '<soc code {}/uuid {}>'.format(self.onet_soc_code, self.uuid)
