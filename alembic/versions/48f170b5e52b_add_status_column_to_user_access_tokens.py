"""add status column to user_access_tokens

Revision ID: 48f170b5e52b
Revises: c3f8a2b1d4e7
Create Date: 2026-02-18 21:12:56.723862

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '48f170b5e52b'
down_revision: Union[str, Sequence[str], None] = 'c3f8a2b1d4e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'user_access_tokens',
        sa.Column('status', sa.String(), nullable=False, server_default='connected'),
    )


def downgrade() -> None:
    op.drop_column('user_access_tokens', 'status')
