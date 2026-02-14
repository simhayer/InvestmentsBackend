"""Persistence layer for chat conversations and messages."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from models.conversation import Conversation, ConversationMessage

logger = logging.getLogger(__name__)


class ConversationStore:
    """Thin wrapper around the conversations / conversation_messages tables."""

    def __init__(self, db: Session, user_id: int):
        self._db = db
        self._user_id = user_id

    # ── create / upsert ─────────────────────────────────────────────

    def ensure_conversation(
        self,
        conversation_id: str,
        *,
        title: Optional[str] = None,
        page_context_type: Optional[str] = None,
    ) -> Conversation:
        """Get or create a conversation row."""
        convo = (
            self._db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.user_id == self._user_id,
            )
            .first()
        )
        if convo:
            if page_context_type:
                convo.page_context_type = page_context_type  # type: ignore[assignment]
            return convo

        convo = Conversation(
            id=conversation_id,
            user_id=self._user_id,
            title=title or "New conversation",
            page_context_type=page_context_type,
        )
        self._db.add(convo)
        self._db.flush()
        return convo

    def add_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMessage:
        """Append a message to a conversation."""
        msg = ConversationMessage(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata_=metadata,
        )
        self._db.add(msg)
        self._db.flush()
        return msg

    # ── read ────────────────────────────────────────────────────────

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return (
            self._db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.user_id == self._user_id,
            )
            .first()
        )

    def list_conversations(self, limit: int = 20, offset: int = 0) -> List[Conversation]:
        return (
            self._db.query(Conversation)
            .filter(Conversation.user_id == self._user_id)
            .order_by(Conversation.updated_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_messages(
        self,
        conversation_id: str,
        *,
        limit: int = 50,
    ) -> List[ConversationMessage]:
        return (
            self._db.query(ConversationMessage)
            .filter(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.asc())
            .limit(limit)
            .all()
        )

    # ── commit helper ───────────────────────────────────────────────

    def commit(self) -> None:
        """Commit the current DB transaction (call after the stream finishes)."""
        try:
            self._db.commit()
        except Exception:
            logger.exception("conversation_store: commit failed")
            self._db.rollback()
