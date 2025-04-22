# my_project/agent_tools.py

import os
import datetime
import logging
import smtplib # Added for CustomerServiceTool
from email.message import EmailMessage # Added for CustomerServiceTool
from enum import Enum # Added for CustomerServiceTool
from typing import Type, Optional, Callable, Set, Dict, ClassVar

# --- Pydantic and Langchain Imports ---
from pydantic import BaseModel, Field
from langchain.vectorstores import FAISS
from langchain.tools import BaseTool

# --- Local Controller/Function Imports ---
from controllers.weather_call import WeatherController
from controllers.tavily import research as tavily_research_func # Rename for clarity

# --- Logger ---
logger = logging.getLogger(__name__)

# --- Control flag for Customer Service Tool ---
STUB_EMAIL_SENDING = True # Set to False for actual email sending
# --- End control flag ---


# ==================================
# Customer Service Tool Components
# ==================================

class ServiceType(str, Enum):
    """Enumeration of supported customer service request types."""
    ROOM_SERVICE = "ROOM_SERVICE"
    DINING_BOOKING = "DINING_BOOKING"
    SPA_BOOKING = "SPA_BOOKING"
    CABANA_BOOKING = "CABANA_BOOKING"

class CustomerServiceInput(BaseModel):
    """Input schema for the CustomerServiceTool."""
    request_type: ServiceType = Field(description="The specific category of the service being requested (e.g., ROOM_SERVICE, DINING_BOOKING).")
    details: str = Field(description="The specific details of the user's request (e.g., 'extra towels', 'table for 2 at 7 PM', 'couples massage tomorrow'). Should include all necessary info gathered.")
    phone_number: str = Field(description="The user's phone number, required for context and processing the request.")
    preferences_summary: Optional[str] = Field(None, description="A summary of the user's known preferences.")


# ==================================
# Input Schemas for Other Tools
# ==================================

class SearchInput(BaseModel):
    query: str = Field(description="The search query relevant to the specific context (facilities, preferences, best practices)")

class PreferenceSearchInput(BaseModel):
    query: str = Field(description="The search query relevant to the user's preferences")
    phone_number: str = Field(description="The user's phone number required to access their specific preferences")

class WeatherInput(BaseModel):
    location: str = Field(description="The location for which to get the weather, typically 'Cancun'")

class WebSearchInput(BaseModel):
    query: str = Field(description="The query for the web search engine")


# ==================================
# Tool Definitions
# ==================================

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
        return self._run()

class FacilitiesSearchTool(BaseTool):
    name: str = "search_resort_facilities"
    description: str = "Useful for finding information about the resort's facilities, amenities, restaurants, locations, opening hours, etc."
    args_schema: Type[BaseModel] = SearchInput
    vector_store: FAISS

    def _run(self, query: str) -> str:
        try:
            docs = self.vector_store.similarity_search(query, k=5)
            if docs:
                return "\n".join(doc.page_content for doc in docs)
            else:
                return "No specific information found about facilities for that query."
        except Exception as e:
            logger.error(f"Error searching facilities: {e}")
            return "Error searching facilities information."

    async def _arun(self, query: str) -> str:
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
    get_user_preferences_store: Callable # Function to get user-specific store

    def _run(self, query: str, phone_number: str) -> str:
        try:
            user_prefs_store = self.get_user_preferences_store(phone_number)
            if not user_prefs_store:
                 return "User preference store not available for searching." # Clarified message
            docs = user_prefs_store.similarity_search(query, k=5)
            if docs:
                return "\n".join(doc.page_content for doc in docs)
            else:
                try: # Check if the index is actually empty vs just no results
                    if user_prefs_store.index.ntotal == 0:
                        return "No preferences have been recorded for this user yet."
                except Exception: # Handle cases where index might not be initialized correctly yet
                     logger.warning(f"Could not check preference store size for {phone_number}")
                return "No specific preferences found matching that query."
        except Exception as e:
            logger.error(f"Error searching preferences for {phone_number}: {e}")
            return "Error searching user preferences."

    async def _arun(self, query: str, phone_number: str) -> str:
        try:
            user_prefs_store = self.get_user_preferences_store(phone_number)
            if not user_prefs_store:
                 return "User preference store not available for searching."
            docs = await user_prefs_store.asimilarity_search(query, k=5)
            if docs:
                return "\n".join(doc.page_content for doc in docs)
            else:
                 try:
                     if user_prefs_store.index.ntotal == 0:
                         return "No preferences have been recorded for this user yet."
                 except Exception:
                     logger.warning(f"Could not check preference store size async for {phone_number}")
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
        # --- Using previous corrected logic ---
        try:
            coords = self.weather_controller.get_coordinates(location)
            if not coords:
                return f"Could not find coordinates for {location}."

            weather_data = self.weather_controller.get_weather(
                coords["latitude"], coords["longitude"],
                hourly=["temperature_2m", "relative_humidity_2m"],
                daily=["temperature_2m_max", "temperature_2m_min", "precipitation_probability_max"],
                forecast_days=3
            )

            if weather_data:
                current = weather_data.get('current', {})
                daily_data = weather_data.get('daily', {})
                temp_c = current.get('temperature_2m', 'N/A')
                humidity = current.get('relative_humidity_2m', 'N/A')
                wind_kmh = current.get('wind_speed_10m', 'N/A')

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
                logger.warning(f"Weather controller returned no data for {location}.")
                return f"Could not retrieve weather data for {location}."
        except Exception as e:
            logger.exception(f"Error getting weather for {location}: {e}")
            return f"Error retrieving weather information for {location}."

    async def _arun(self, location: str = "Cancun") -> str:
        return self._run(location)

class TavilySearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Useful for answering questions about general knowledge, current events, external services, or information not specific to the resort itself."
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        # --- Using previous corrected logic ---
        try:
            tavily_result = tavily_research_func(query, include_sources=False)
            if tavily_result.get('status') == 'success' and tavily_result.get('answer'):
                return f"Web Search Result: {tavily_result['answer']}"
            elif tavily_result.get('status') == 'error':
                 logger.warning(f"Tavily search failed for query '{query}': {tavily_result.get('error')}")
                 return "Web search failed."
            else:
                return "No relevant information found from web search."
        except Exception as e:
            logger.error(f"Error calling Tavily research for query '{query}': {e}")
            return "Error during web search."

    async def _arun(self, query: str) -> str:
        return self._run(query)


# ==================================
# Customer Service Tool Definition (Integrated)
# ==================================
class CustomerServiceTool(BaseTool):
    """
    Tool to handle specific guest service requests like Room Service, Dining Reservations, Spa Bookings, and Cabana inquiries.
    It attempts to forward the request to the relevant department via email (or simulates it) for certain types,
    and provides specific instructions for others (like Cabana bookings).
    **IMPORTANT:** This tool should ONLY be called AFTER the necessary details have been gathered through conversation.
    """
    name: str = "handle_customer_service_request"
    description: str = (
        "Use this tool ONLY AFTER gathering ALL required details for specific guest service requests like: "
        "Room Service (food, towels, amenities), Dining Reservations (needs date, time, party size), "
        "Spa Bookings (needs date, preferred time/service), or Cabana inquiries/bookings (needs date). "
        "DO NOT use this tool if details are missing - ask clarifying questions first. "
        "Once ALL details are confirmed, provide the 'request_type' (ROOM_SERVICE, DINING_BOOKING, SPA_BOOKING, CABANA_BOOKING), "
        "the complete 'details' gathered, the user's 'phone_number', and optional 'preferences_summary'."
    )
    args_schema: Type[BaseModel] = CustomerServiceInput

    # --- SMTP Configuration ---
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    recipient_email: Optional[str] = None
    smtp_configured: bool = False

    # ---> FIX: Add ClassVar annotations <---
    EMAIL_SERVICE_TYPES: ClassVar[Set[ServiceType]] = {
        ServiceType.ROOM_SERVICE,
        ServiceType.DINING_BOOKING,
        ServiceType.SPA_BOOKING
    }
    INSTRUCTION_SERVICE_TYPES: ClassVar[Dict[ServiceType, str]] = {
        ServiceType.CABANA_BOOKING: "For Cabana bookings, please call our Pool Concierge directly at 1-800-CABANAS (example number)."
    }
    # ---> End FIX <---

    # The rest of the __init__, _run, helper methods remain the same...
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ... (SMTP init logic remains the same) ...
        self.smtp_server = os.getenv("SMTP_SERVER")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.recipient_email = os.getenv("CUSTOMER_SERVICE_EMAIL")
        self.smtp_configured = all([self.smtp_server, self.smtp_port, self.smtp_user, self.smtp_password, self.recipient_email])

        if not self.smtp_configured:
            logger.warning("CustomerServiceTool: SMTP config incomplete. Emails disabled/stubbed (STUB_EMAIL_SENDING=%s).", STUB_EMAIL_SENDING)
        else:
            logger.info(f"CustomerServiceTool initialized. Emails to {self.recipient_email}. STUB_EMAIL_SENDING={STUB_EMAIL_SENDING}")

    # ... (_send_request_email, _generate_fulfillment_response, _generate_cannot_fulfill_response, _run, _arun methods remain the same) ...
    def _send_request_email(self, request_type: str, message_details: str, user_phone: str, prefs_summary: Optional[str]) -> bool:
        """Internal method to send email or stub it."""
        if STUB_EMAIL_SENDING:
            logger.info(f"--- STUBBED EMAIL (CustomerServiceTool) ---")
            logger.info(f"To: {self.recipient_email}, Type: {request_type}, Phone: {user_phone}")
            logger.info(f"Prefs: {prefs_summary or 'None'}")
            logger.info(f"Details: {message_details}")
            logger.info("Request sent to guest services Successfully (Simulated)")
            return True

        if not self.smtp_configured:
            logger.error("SMTP is not configured. Cannot send email.")
            return False

        subject = f"New {request_type.replace('_', ' ').title()} Request from {user_phone}"
        email_content = f"Guest Phone: {user_phone}\nRequest Type: {request_type}\n\nPreferences:\n{prefs_summary or 'None Recorded'}\n\nDetails:\n{message_details}\n\nPlease process this request."
        msg = EmailMessage()
        msg.set_content(email_content)
        msg['Subject'] = subject
        msg['From'] = self.smtp_user
        msg['To'] = self.recipient_email

        try:
            logger.info(f"Attempting to send {request_type} email to {self.recipient_email}")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            logger.info(f"Successfully sent {request_type} email for {user_phone}.")
            return True
        except Exception as e:
            logger.error(f"Failed to send SMTP email for {request_type} from {user_phone}: {e}", exc_info=True)
            return False

    def _generate_fulfillment_response(self, request_type: ServiceType, details: str) -> str:
        """Generates standard confirmation for email-based requests."""
        # Use f-strings for cleaner formatting
        base_responses = {
            ServiceType.ROOM_SERVICE: f"Okay, I've passed your room service request ({details}) to our team.^MSG^They'll process it shortly.",
            ServiceType.DINING_BOOKING: f"Thanks! I've forwarded your dining reservation request ({details}).^MSG^Someone will confirm availability with you if needed.",
            ServiceType.SPA_BOOKING: f"Got it. I've sent your spa booking request ({details}) to the wellness center.^MSG^They'll reach out to confirm details.",
        }
        return base_responses.get(request_type, f"Okay, I'm processing your request ({details}).")

    def _generate_cannot_fulfill_response(self, request_type: ServiceType, details: str, reason: str = "system issue") -> str:
        """Generates message indicating request cannot be fulfilled automatically."""
        if reason == "smtp_disabled": issue = "our notification system is currently unavailable"
        elif reason == "smtp_failed": issue = "there was a problem sending your request to the team"
        else: issue = "I encountered an issue processing that automatically"

        # Use f-strings
        base_message = f"I understand you'd like help with {request_type.value.replace('_', ' ').lower()} ({details}). However, {issue}."
        suggestion = "Could you please call the front desk or concierge directly for assistance?"
        return f"{base_message}^MSG^{suggestion}"

    def _run(self, request_type: ServiceType, details: str, phone_number: str, preferences_summary: Optional[str] = None) -> str:
        """Executes the tool's logic based on the request type."""
        logger.info(f"CustomerServiceTool invoked: type='{request_type}', details='{details}', phone='{phone_number}'")

        if request_type in self.EMAIL_SERVICE_TYPES:
            if not self.smtp_configured and not STUB_EMAIL_SENDING:
                logger.warning(f"Cannot fulfill {request_type} for {phone_number} due to SMTP config/stubbing.")
                return self._generate_cannot_fulfill_response(request_type, details, reason="smtp_disabled")

            email_sent_or_simulated = self._send_request_email(request_type.value, details, phone_number, preferences_summary)

            if email_sent_or_simulated:
                return self._generate_fulfillment_response(request_type, details)
            else: # Only reached if STUBBING is False and send failed
                return self._generate_cannot_fulfill_response(request_type, details, reason="smtp_failed")

        elif request_type in self.INSTRUCTION_SERVICE_TYPES:
            instruction = self.INSTRUCTION_SERVICE_TYPES[request_type]
            logger.info(f"Providing instructions for {request_type} for {phone_number}.")
            return f"Okay, regarding your request for {request_type.value.replace('_', ' ').lower()} ({details}):^MSG^{instruction}"

        else: # Should not happen if agent follows description
            logger.warning(f"CustomerServiceTool received unsupported request_type: {request_type}")
            return f"I understand you're asking about {details}, but I encountered an issue with the request type '{request_type.value}'. Please contact the front desk."

    async def _arun(self, request_type: ServiceType, details: str, phone_number: str, preferences_summary: Optional[str] = None) -> str:
        # If _send_request_email becomes async, update this
        return self._run(request_type, details, phone_number, preferences_summary)