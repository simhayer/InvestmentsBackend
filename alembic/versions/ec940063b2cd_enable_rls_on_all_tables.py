"""enable rls on all tables

Revision ID: ec940063b2cd
Revises: 48f170b5e52b
Create Date: 2026-02-18
"""
from typing import Sequence, Union
from alembic import op

revision: str = 'ec940063b2cd'
down_revision: Union[str, Sequence[str], None] = '48f170b5e52b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ALL_TABLES = [
    "users",
    "holdings",
    "user_access_tokens",
    "user_subscriptions",
    "user_onboarding_profiles",
    "conversations",
    "conversation_messages",
    "portfolio_analyses",
    "market_summaries",
    "market_overview_latest",
    "market_overview_history",
    "crypto_assets",
    "sec_filing_chunks",
    "company_insights",
]


def upgrade() -> None:
    for table in ALL_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f'CREATE POLICY "deny_all_{table}" ON {table} '
            f"FOR ALL TO anon, authenticated USING (false)"
        )


def downgrade() -> None:
    for table in ALL_TABLES:
        op.execute(f'DROP POLICY IF EXISTS "deny_all_{table}" ON {table}')
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
