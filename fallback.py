from random import choice


class FallbackConversation:

    def __init__(self):

        self.messages = [

            "That's an interesting question. Give me a moment while I think.",

            "Let me analyze that for you.",

            "I'm processing your request. This will only take a moment.",

            "Thinking... I'll give you the best answer I can.",

            "Let me gather my thoughts before I answer.",

            "Please wait while I generate a detailed response."

        ]

    def get_message(self):

        return choice(self.messages)