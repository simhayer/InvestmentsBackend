"""add refund columns to user_subscriptions

Revision ID: b7d3e9f1a2c4
Revises: ec940063b2cd
Create Date: 2026-02-21

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b7d3e9f1a2c4"
down_revision: Union[str, None] = "ec940063b2cd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user_subscriptions",
        sa.Column("has_used_refund", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "user_subscriptions",
        sa.Column("refunded_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_subscriptions", "refunded_at")
    op.drop_column("user_subscriptions", "has_used_refund")
