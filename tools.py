# my_project/tools.py

import datetime
import logging
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from controllers.weather_call import WeatherController
from controllers.tavily import research # Assuming this function exists
from models.file_storage_helper import FileStorageHelper # Assuming this exists
from models.spam_filter import SpamFilter # Assuming this exists
from twilio.rest import Client
import time

logger = logging.getLogger(__name__)

# --- Context Retrieval Tools ---

class ContextRetrieverTool:
    def __init__(self, facilities_vector_store: FAISS,
                 best_practices_vector_store: FAISS,
                 get_user_preferences_store, # Function to get user-specific store
                 weather_controller: WeatherController,
                 tavily_research_func): # Pass the research function
        self.facilities_vector_store = facilities_vector_store
        self.best_practices_vector_store = best_practices_vector_store
        self.get_user_preferences_store = get_user_preferences_store
        self.weather_controller = weather_controller
        self.tavily_research_func = tavily_research_func

    def run(self, user_query: str, phone_number: str) -> str:
        """Retrieves comprehensive context for a user query."""
        context_parts = []

        # Add current Cancun time (UTC-5)
        try:
            current_time_utc = datetime.datetime.utcnow()
            cancun_time = current_time_utc - datetime.timedelta(hours=5)
            context_parts.append(f"Current Cancun Date/Time: {cancun_time.strftime('%Y-%m-%d %H:%M:%S')} (UTC-5)")
        except Exception as e:
            logger.error(f"Error getting current time: {e}")

        # 1) Facilities-based retrieval
        try:
            fac_docs = self.facilities_vector_store.similarity_search(user_query, k=5)
            if fac_docs:
                facilities_info = "\n".join(doc.page_content for doc in fac_docs)
                context_parts.append(f"Facilities Info:\n{facilities_info}")
        except Exception as e:
            logger.error(f"Error retrieving facilities info: {e}")

        # 2) Preference-based retrieval (unique to the user)
        try:
            user_prefs_store = self.get_user_preferences_store(phone_number)
            if user_prefs_store: # Check if store exists/was created
                 pref_docs = user_prefs_store.similarity_search(user_query, k=5)
                 if pref_docs:
                     preferences_info = "\n".join(doc.page_content for doc in pref_docs)
                     context_parts.append(f"Preferences Info:\n{preferences_info}")
        except Exception as e:
            logger.error(f"Error retrieving preferences info for {phone_number}: {e}")


        # 3) Best practices-based retrieval
        try:
            bp_docs = self.best_practices_vector_store.similarity_search(user_query, k=2)
            if bp_docs:
                bp_info = "\n".join(doc.page_content for doc in bp_docs)
                context_parts.append(f"Best Practices Info:\n{bp_info}")
        except Exception as e:
            logger.error(f"Error retrieving best practices info: {e}")

        # 4) Add weather context for Cancun
        try:
            coords = self.weather_controller.get_coordinates("Cancun")
            if coords:
                weather_data = self.weather_controller.get_weather(
                    coords["latitude"], coords["longitude"],
                    hourly=["temperature_2m", "relative_humidity_2m"],
                    daily=["temperature_2m_max", "temperature_2m_min", "precipitation_probability_max"],
                    forecast_days=8
                )
                if weather_data:
                    weather_str = (
                        f"Current Cancun Weather:\n"
                        f"- Temperature: {weather_data['current']['temperature_2m']}°C\n"
                        f"- Wind Speed: {weather_data['current']['wind_speed_10m']} km/h\n"
                        f"Today's Hourly Forecast:\n" + "\n".join(
                            f"{time.split('T')[1][:5]}: {temp}°C, {humidity}% humidity"
                            for time, temp, humidity in zip(
                                weather_data['hourly']['time'][:24],
                                weather_data['hourly']['temperature_2m'][:24],
                                weather_data['hourly']['relative_humidity_2m'][:24]
                            )
                        ) + "\n\n7-Day Forecast:\n" + "\n".join(
                            f"{date}: {tmax}°C/{tmin}°C, {rain}% rain"
                            for date, tmax, tmin, rain in zip(
                                weather_data['daily']['time'][1:8],
                                weather_data['daily']['temperature_2m_max'][1:8],
                                weather_data['daily']['temperature_2m_min'][1:8],
                                weather_data['daily']['precipitation_probability_max'][1:8]
                            )
                        )
                    )
                    context_parts.append(weather_str)
        except Exception as e:
            logger.error(f"Error retrieving weather info: {e}")

        # 5) Add Tavily web search results
        try:
            tavily_result = self.tavily_research_func(user_query, include_sources=True)
            if tavily_result.get('status') == 'success' and tavily_result.get('answer'):
                tavily_info = f"Web Search Results (from Tavily):\nAnswer: {tavily_result['answer']}"
                if tavily_result.get('sources'):
                    sources_str = "\nSources:\n" + "\n".join(
                        f"- {source.get('title', 'N/A')}: {source.get('url', 'N/A')}"
                        for source in tavily_result['sources'][:3] # Limit to top 3 sources
                    )
                    tavily_info += sources_str
                context_parts.append(tavily_info)
            elif tavily_result.get('status') == 'error':
                 logger.warning(f"Tavily search failed for query '{user_query}': {tavily_result.get('error')}")
        except Exception as e:
            logger.error(f"Error calling Tavily research for query '{user_query}': {str(e)}")

        if not context_parts:
            return "No relevant context found."

        # print("--- Retrieved Context ---") # Optional Debugging
        # print("\n\n".join(context_parts))
        # print("--- End Context ---")
        return "\n\n".join(context_parts)

# --- Preference Management Tools ---

class PreferenceLoaderTool:
    def __init__(self, file_storage_helper: FileStorageHelper, get_prefs_filename):
        self.file_storage_helper = file_storage_helper
        self.get_prefs_filename = get_prefs_filename

    def run(self, phone_number: str) -> str:
        """Loads raw customer preferences text."""
        filename = self.get_prefs_filename(phone_number)
        return "\n".join(self.file_storage_helper.load_lines(filename))

class PreferenceUpdateTool:
    """Uses an LLM to extract and update preferences based on conversation."""
    def __init__(self, llm, get_prefs_filename, file_storage_helper, get_user_preferences_store):
        self.llm = llm
        self.get_prefs_filename = get_prefs_filename
        self.file_storage_helper = file_storage_helper
        self.get_user_preferences_store = get_user_preferences_store
        # Simplified prompt for preference extraction
        self.prompt = PromptTemplate(
            input_variables=["history", "current_preferences"],
            template="""
            Analyze the following conversation history and current preferences.
            Extract any new or updated preferences mentioned by the user.
            List *only* the key preference points, one per line. If no new preferences, output "None".

            Current Preferences:
            {current_preferences}

            Conversation History:
            {history}

            Extracted/Updated Preferences:
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, verbose=False)

    def run(self, phone_number: str, history: str):
        """Analyzes history, updates preference file and vector store."""
        filename = self.get_prefs_filename(phone_number)
        current_prefs_text = "\n".join(self.file_storage_helper.load_lines(filename))

        # Use LLM to extract potential new preferences
        extracted_prefs_str = self.chain.run(history=history, current_preferences=current_prefs_text)

        new_prefs_list = [line.strip() for line in extracted_prefs_str.split('\n') if line.strip() and line.strip().lower() != "none"]

        if not new_prefs_list:
            logger.info(f"No new preferences detected for {phone_number}.")
            return # Nothing to update

        logger.info(f"Updating preferences for {phone_number} with: {new_prefs_list}")

        # Append new preferences to the file
        self.file_storage_helper.append_lines(filename, new_prefs_list)

        # Update the vector store
        try:
            user_prefs_store = self.get_user_preferences_store(phone_number)
            if user_prefs_store:
                 # Ensure texts are non-empty before adding
                 valid_new_prefs = [pref for pref in new_prefs_list if pref]
                 if valid_new_prefs:
                    user_prefs_store.add_texts(valid_new_prefs)
                 else:
                    logger.warning(f"Attempted to add empty preferences for {phone_number}")

        except Exception as e:
            logger.error(f"Failed to update preference vector store for {phone_number}: {e}")

# --- Utility Tools ---

class SpamFilterTool:
    def __init__(self, spam_filter: SpamFilter):
        self.spam_filter = spam_filter

    def run(self, message: str) -> bool:
        """Checks if a message is spam."""
        return self.spam_filter.is_blocked_message(message)

class MessageSenderTool:
    def __init__(self, twilio_client: Client, from_number: str):
        self.twilio_client = twilio_client
        self.from_number = from_number

    def _compute_typing_delay(self, message: str, wpm: int = 100) -> float:
        word_count = len(message.split())
        seconds_per_word = 60.0 / wpm # Corrected calculation
        delay = word_count * seconds_per_word
        return min(delay, 1.6) # Cap delay

    def run(self, to_number: str, message_body: str):
        """Sends a message via Twilio, handling multi-bubble and delay."""
        segments = message_body.split("^MSG^")
        first_segment = True
        for seg in segments:
            cleaned = seg.strip()
            if cleaned:
                if first_segment:
                    logger.info(f"Sending WhatsApp to={to_number}, body='{cleaned}'")
                    self.twilio_client.messages.create(
                        from_=self.from_number,
                        body=cleaned,
                        to=to_number
                    )
                    first_segment = False
                else:
                    delay = self._compute_typing_delay(cleaned, wpm=100)
                    logger.info(f"Sleeping for {delay:.2f} seconds before sending next part.")
                    time.sleep(delay)
                    logger.info(f"Sending WhatsApp to={to_number}, body='{cleaned}'")
                    self.twilio_client.messages.create(
                        from_=self.from_number,
                        body=cleaned,
                        to=to_number
                    )