"""
history.py — Conversation history logging.

Stage 1 fixes applied:
  D-6  — replaced bare print() with logging
  M-5  — history directory from config.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import config

logger = logging.getLogger(__name__)


class ConversationHistory:
    """Appends each user/assistant turn to a session JSON file."""

    def __init__(self) -> None:
        history_dir: Path = config.HISTORY_DIR
        history_dir.mkdir(parents=True, exist_ok=True)

        self.filename: Path = history_dir / (
            datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        )
        self.messages: List[Dict[str, str]] = []
        logger.info("Conversation history → %s", self.filename)

    def save(self, user: str, assistant: str) -> None:
        """Append a turn and flush to disk."""
        self.messages.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "user": user,
                "assistant": assistant,
            }
        )
        with open(self.filename, "w", encoding="utf-8") as file:
            json.dump(self.messages, file, indent=4, ensure_ascii=False)
        logger.debug("History saved (%d turns).", len(self.messages))

    def get_history_for_llm(self) -> List[Dict[str, str]]:
        """
        Return accumulated conversation turns formatted for Ollama:
        [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        """
        formatted: List[Dict[str, str]] = []
        for turn in self.messages:
            user_msg = turn.get("user", "").strip()
            assistant_msg = turn.get("assistant", "").strip()
            if user_msg:
                formatted.append({"role": "user", "content": user_msg})
            if assistant_msg:
                formatted.append({"role": "assistant", "content": assistant_msg})
        return formatted

    def clear(self) -> None:
        """Clear memory for current session."""
        self.messages.clear()
        logger.debug("Conversation history memory cleared.")