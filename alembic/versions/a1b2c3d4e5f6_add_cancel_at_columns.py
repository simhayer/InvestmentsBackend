"""add cancel_at columns to user_subscriptions

Revision ID: a1b2c3d4e5f6
Revises: 799e61673527
Create Date: 2026-02-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "799e61673527"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user_subscriptions",
        sa.Column("cancel_at_period_end", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "user_subscriptions",
        sa.Column("cancel_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_subscriptions", "cancel_at")
    op.drop_column("user_subscriptions", "cancel_at_period_end")
