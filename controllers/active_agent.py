# my_project/controllers/active_agent.py

class ActiveAgent:
    """
    A stub or example of a proactive messaging agent.
    This agent could be extended to:
      - Follow up based on user inquiries or preferences
      - Offer room upgrades or upsell services
      - Send periodic reminders or special offers
    """
    def __init__(self):
        # If needed, you could pass additional data or configuration here
        pass

    def proactive_message(self, conversation_history):
        """
        Returns a short proactive message based on the conversation history, or
        an empty string if no proactive message is needed.

        In a real scenario, this might:
          - analyze conversation history for triggers (e.g., user mention of special events),
          - check an internal CRM or reservation system for upcoming birthdays, anniversaries, etc.
          - present relevant offers or reminders.
        """
        # For now, just a stub. You can replace this with actual logic or data.
        return "By the way, feel free to ask about our spa specials and VIP tours!"
