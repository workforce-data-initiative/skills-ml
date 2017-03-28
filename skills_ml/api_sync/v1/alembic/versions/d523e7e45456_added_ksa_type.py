"""Added ksa_type

Revision ID: d523e7e45456
Revises: 
Create Date: 2017-02-27 15:21:53.018866

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd523e7e45456'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('skills_master', sa.Column('ksa_type', sa.String))


def downgrade():
    op.drop_column('skills_master', 'ksa_type')
