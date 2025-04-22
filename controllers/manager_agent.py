# my_project/controllers/manager_agent.py

class ManagerAgent:
    """
    Coordinates the profiling and the front-end agent's conversation logic.
    """
    def __init__(self, profiler_agent, front_agent, active_agent):
        self.profiler = profiler_agent
        self.front_agent = front_agent
        self.active_agent = active_agent

    def handle_interaction(self, phone_number, user_input, conversation_history_str):
        response = self.front_agent.generate_response(
            user_query=user_input,
            history_str=conversation_history_str,
            phone_number=phone_number
        )
        proactive_message = self.active_agent.proactive_message(conversation_history_str)
        return response, proactive_message

    def finalize_preferences(self, phone_number, conversation_history_str):
        new_preferences = self.profiler.profile_user(phone_number, conversation_history_str)
        return new_preferences
