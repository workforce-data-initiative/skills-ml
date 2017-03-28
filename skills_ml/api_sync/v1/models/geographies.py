# -*- coding: utf-8 -*-

from . import Base
from sqlalchemy import Column, SmallInteger, String


class Geography(Base):
    __tablename__ = 'geographies'

    geography_id = Column(SmallInteger, primary_key=True)
    geography_type = Column(String, nullable=False)
    geography_name = Column(String, nullable=False)

    def __repr__(self):
        return '<geography {}/{}>'.format(
            self.geography_type, self.geography_name
        )
