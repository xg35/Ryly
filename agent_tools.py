# my_project/agent_tools.py

import datetime
import logging
from typing import Type, Optional
from pydantic import BaseModel, Field

from langchain.vectorstores import FAISS
from langchain.tools import BaseTool # Import BaseTool
from controllers.weather_call import WeatherController
from controllers.tavily import research as tavily_research_func # Rename for clarity
from models.file_storage_helper import FileStorageHelper

logger = logging.getLogger(__name__)

# --- Input Schemas for Tools ---

class SearchInput(BaseModel):
    query: str = Field(description="The search query relevant to the specific context (facilities, preferences, best practices)")

class PreferenceSearchInput(BaseModel):
    query: str = Field(description="The search query relevant to the user's preferences")
    phone_number: str = Field(description="The user's phone number required to access their specific preferences")

class WeatherInput(BaseModel):
    location: str = Field(description="The location for which to get the weather, typically 'Cancun'")

class WebSearchInput(BaseModel):
    query: str = Field(description="The query for the web search engine")

# --- Granular Tool Definitions ---

class GetCurrentTimeTool(BaseTool):
    name: str = "get_current_cancun_time"
    description: str = "Useful for getting the current date and time in Cancun (UTC-5)."

    def _run(self) -> str:
        try:
            current_time_utc = datetime.datetime.utcnow()
            cancun_time = current_time_utc - datetime.timedelta(hours=5)
            return f"Current Cancun Date/Time: {cancun_time.strftime('%Y-%m-%d %H:%M:%S')} (UTC-5)"
        except Exception as e:
            logger.error(f"Error getting current time: {e}")
            return "Error getting current time."

    async def _arun(self) -> str:
        # Implement async version if needed, otherwise fallback
        return self._run()

class FacilitiesSearchTool(BaseTool):
    name: str = "search_resort_facilities"
    description: str = "Useful for finding information about the resort's facilities, amenities, restaurants, locations, opening hours, etc."
    args_schema: Type[BaseModel] = SearchInput
    vector_store: FAISS

    def _run(self, query: str) -> str:
        try:
            docs = self.vector_store.similarity_search(query, k=5) # Reduced k for agent use
            if docs:
                return "\n".join(doc.page_content for doc in docs)
            else:
                return "No specific information found about facilities for that query."
        except Exception as e:
            logger.error(f"Error searching facilities: {e}")
            return "Error searching facilities information."

    async def _arun(self, query: str) -> str:
        # Implement async version if needed
        try:
            docs = await self.vector_store.asimilarity_search(query, k=5)
            if docs:
                return "\n".join(doc.page_content for doc in docs)
            else:
                return "No specific information found about facilities for that query."
        except Exception as e:
            logger.error(f"Error searching facilities async: {e}")
            return "Error searching facilities information."


class PreferencesSearchTool(BaseTool):
    name: str = "search_user_preferences"
    description: str = "Useful for retrieving the user's known preferences, likes, dislikes, past requests, or dietary restrictions. Requires the user's phone_number."
    args_schema: Type[BaseModel] = PreferenceSearchInput
    get_user_preferences_store: callable # Function to get user-specific store

    def _run(self, query: str, phone_number: str) -> str:
        try:
            user_prefs_store = self.get_user_preferences_store(phone_number)
            if not user_prefs_store:
                 return "User preference store not available."
            docs = user_prefs_store.similarity_search(query, k=5) # Reduced k
            if docs:
                return "\n".join(doc.page_content for doc in docs)
            else:
                # Distinguish between no store and no results
                if user_prefs_store.index.ntotal == 0:
                    return "No preferences have been recorded for this user yet."
                else:
                    return "No specific preferences found matching that query."
        except Exception as e:
            logger.error(f"Error searching preferences for {phone_number}: {e}")
            return "Error searching user preferences."

    async def _arun(self, query: str, phone_number: str) -> str:
         # Implement async version if needed
        try:
            user_prefs_store = self.get_user_preferences_store(phone_number) # Assuming this is sync for now
            if not user_prefs_store:
                 return "User preference store not available."
            docs = await user_prefs_store.asimilarity_search(query, k=5)
            if docs:
                return "\n".join(doc.page_content for doc in docs)
            else:
                 if user_prefs_store.index.ntotal == 0:
                    return "No preferences have been recorded for this user yet."
                 else:
                    return "No specific preferences found matching that query."
        except Exception as e:
            logger.error(f"Error searching preferences async for {phone_number}: {e}")
            return "Error searching user preferences."


class BestPracticesSearchTool(BaseTool):
    name: str = "search_resort_best_practices"
    description: str = "Useful for finding general advice, tips, or best practices related to the resort experience (e.g., booking recommendations, general policies)."
    args_schema: Type[BaseModel] = SearchInput
    vector_store: FAISS

    def _run(self, query: str) -> str:
        try:
            docs = self.vector_store.similarity_search(query, k=2)
            if docs:
                return "\n".join(doc.page_content for doc in docs)
            else:
                return "No specific best practices found for that query."
        except Exception as e:
            logger.error(f"Error searching best practices: {e}")
            return "Error searching best practices."

    async def _arun(self, query: str) -> str:
        # Implement async version if needed
        try:
            docs = await self.vector_store.asimilarity_search(query, k=2)
            if docs:
                return "\n".join(doc.page_content for doc in docs)
            else:
                return "No specific best practices found for that query."
        except Exception as e:
            logger.error(f"Error searching best practices async: {e}")
            return "Error searching best practices."

class WeatherTool(BaseTool):
    name: str = "get_cancun_weather"
    description: str = "Useful for getting the current weather conditions and forecast for Cancun. Use this when the user asks about weather or activities that depend on it (beach, pool, outdoor excursions)."
    args_schema: Type[BaseModel] = WeatherInput
    weather_controller: WeatherController

    def _run(self, location: str = "Cancun") -> str:
        try:
            coords = self.weather_controller.get_coordinates(location)
            if not coords:
                return f"Could not find coordinates for {location}."

            # --- FIX: Removed the 'current' keyword argument ---
            weather_data = self.weather_controller.get_weather(
                coords["latitude"], coords["longitude"],
                hourly=["temperature_2m", "relative_humidity_2m"], # Still request hourly
                daily=["temperature_2m_max", "temperature_2m_min", "precipitation_probability_max"],
                # current=[...] <--- REMOVED THIS LINE
                forecast_days=3
            )
            # ----------------------------------------------------

            # The rest of the logic with .get() remains the same for safety
            if weather_data:
                current = weather_data.get('current', {}) # Get current dict safely
                daily_data = weather_data.get('daily', {}) # Get daily dict safely

                # Access current conditions safely
                temp_c = current.get('temperature_2m', 'N/A')
                humidity = current.get('relative_humidity_2m', 'N/A')
                wind_kmh = current.get('wind_speed_10m', 'N/A')

                # Format forecast safely
                forecast_lines = []
                daily_times = daily_data.get('time', [])
                daily_max_temps = daily_data.get('temperature_2m_max', [])
                daily_min_temps = daily_data.get('temperature_2m_min', [])
                daily_precip_probs = daily_data.get('precipitation_probability_max', [])

                num_days = min(len(daily_times), 3)
                for i in range(num_days):
                    date = daily_times[i]
                    tmax = daily_max_temps[i] if i < len(daily_max_temps) else 'N/A'
                    tmin = daily_min_temps[i] if i < len(daily_min_temps) else 'N/A'
                    rain = daily_precip_probs[i] if i < len(daily_precip_probs) else 'N/A'
                    forecast_lines.append(f"- {date}: Max {tmax}°C, Min {tmin}°C, Rain: {rain}%")

                summary = (
                    f"Current Weather in {location}:\n"
                    f"- Temp: {temp_c}°C, Humidity: {humidity}%, Wind: {wind_kmh} km/h\n"
                    f"Forecast:\n" + "\n".join(forecast_lines)
                )
                return summary
            else:
                logger.warning(f"Weather controller returned no data for {location}. Response: {weather_data}")
                return f"Could not retrieve weather data for {location}."
        except Exception as e:
            logger.exception(f"Error getting weather for {location}: {type(e).__name__}: {str(e)}")
            return f"Error retrieving weather information for {location}."

    async def _arun(self, location: str = "Cancun") -> str:
        return self._run(location)

class TavilySearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Useful for answering questions about general knowledge, current events, external services, or information not specific to the resort itself."
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        try:
            # --- FIX: Removed max_results=3 ---
            tavily_result = tavily_research_func(query, include_sources=False)
            # ---------------------------------

            # print(f"DEBUG: Tavily Result for query '{query}': {tavily_result}") # Keep for debugging if needed
            if tavily_result.get('status') == 'success' and tavily_result.get('answer'):
                # Limit result length here if needed, e.g., return f"Web Search Result: {tavily_result['answer'][:500]}"
                return f"Web Search Result: {tavily_result['answer']}"
            elif tavily_result.get('status') == 'error':
                 logger.warning(f"Tavily search failed for query '{query}': {tavily_result.get('error')}")
                 return "Web search failed."
            else:
                return "No relevant information found from web search."
        except Exception as e:
            # Log the specific exception type and message
            logger.error(f"Error calling Tavily research for query '{query}': {type(e).__name__}: {str(e)}")
            return "Error during web search."

    async def _arun(self, query: str) -> str:
        # Async version would depend on tavily having an async interface
        return self._run(query)

# --- Tools related to Preference Management (Called by Orchestrator, not Agent) ---
# Keep PreferenceLoaderTool and PreferenceUpdateTool as defined in the previous step
# in 'tools.py' or move them here if preferred, but they likely won't be given
# directly to the main response-generating agent.

# --- Tools related to Messaging (Called by Orchestrator, not Agent) ---
# Keep SpamFilterTool and MessageSenderTool as defined previously.