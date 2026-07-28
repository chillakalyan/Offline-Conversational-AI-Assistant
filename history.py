import json
import os
from datetime import datetime


class ConversationHistory:

    def __init__(self):

        os.makedirs("history", exist_ok=True)

        self.filename = (
            "history/"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".json"
        )

        self.messages = []

    def save(self, user, assistant):

        self.messages.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "user": user,
                "assistant": assistant
            }
        )

        with open(self.filename, "w", encoding="utf-8") as file:

            json.dump(
                self.messages,
                file,
                indent=4,
                ensure_ascii=False
            )