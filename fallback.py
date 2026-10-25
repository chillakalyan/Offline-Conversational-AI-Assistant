"""
fallback.py — Random filler phrases spoken while the LLM is thinking.

Stage 1 fixes applied:
  D-6  — replaced bare print() (none existed here) with logging for
          visibility; no logic change.
"""

import logging
from random import choice
from typing import List

logger = logging.getLogger(__name__)


class FallbackConversation:
    """Returns a random filler phrase to keep the user engaged while the LLM generates."""

    def __init__(self) -> None:
        self.messages: List[str] = [
            "That's an interesting question. Give me a moment while I think.",
            "Let me analyze that for you.",
            "I'm processing your request. This will only take a moment.",
            "Thinking… I'll give you the best answer I can.",
            "Let me gather my thoughts before I answer.",
            "Please wait while I generate a detailed response.",
        ]

    def get_message(self) -> str:
        message = choice(self.messages)
        logger.debug("Fallback message selected: %s", message)
        return message