# -*- coding: utf-8 -*-

import sqlalchemy as db
from . import Base


class Quarter(Base):
    __tablename__ = 'quarters'

    quarter_id = db.Column(db.SmallInteger, primary_key=True)
    year = db.Column(db.Integer, nullable=False)
    quarter = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return '<Quarter {}/{}>'.format(
            self.year, self.quarter
        )
