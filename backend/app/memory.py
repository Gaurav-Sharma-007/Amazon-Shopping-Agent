from __future__ import annotations

import json
import uuid
from pathlib import Path
from threading import Lock

from .config import get_settings
from .models import ChatMessage, ProductFilters


class SessionMemory:
    """Small JSON-backed session store for continuous conversation context."""

    def __init__(self) -> None:
        settings = get_settings()
        settings.local_data_dir.mkdir(parents=True, exist_ok=True)
        self._path = settings.local_data_dir / "sessions.json"
        self._lock = Lock()

    def create_session_id(self) -> str:
        return uuid.uuid4().hex

    def load(self, session_id: str) -> dict:
        with self._lock:
            data = self._read_all()
            return data.get(
                session_id,
                {
                    "messages": [],
                    "filters": ProductFilters().model_dump(),
                    "last_recommendation": None,
                },
            )

    def save(
        self,
        session_id: str,
        messages: list[ChatMessage],
        filters: ProductFilters,
        recommendation: dict | None,
    ) -> None:
        with self._lock:
            data = self._read_all()
            data[session_id] = {
                "messages": [message.model_dump() for message in messages],
                "filters": filters.model_dump(),
                "last_recommendation": recommendation,
            }
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _read_all(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

