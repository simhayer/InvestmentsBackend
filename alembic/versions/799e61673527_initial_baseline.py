"""initial baseline

Revision ID: 799e61673527
Revises: 
Create Date: 2026-02-13 00:42:14.252106

This is a no-op baseline migration. The database already existed before
Alembic was introduced. We stamp HEAD to this revision and all future
changes go through proper migrations.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '799e61673527'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op — existing schema adopted as baseline."""
    pass


def downgrade() -> None:
    """No-op — cannot downgrade past the baseline."""
    pass
