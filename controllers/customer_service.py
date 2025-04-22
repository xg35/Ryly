# my_project/customer_service_tool.py (Recommended new filename)

import os
import smtplib
import logging
from email.message import EmailMessage
from enum import Enum
from typing import Type, Optional

# Langchain imports
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

# --- Control flag for stubbing ---
STUB_EMAIL_SENDING = True # Set to False for actual email sending
# --- End control flag ---

# --- Define Service Types ---
class ServiceType(str, Enum):
    """Enumeration of supported customer service request types."""
    ROOM_SERVICE = "ROOM_SERVICE"
    DINING_BOOKING = "DINING_BOOKING"
    SPA_BOOKING = "SPA_BOOKING"
    CABANA_BOOKING = "CABANA_BOOKING"
    # Add other types here if needed

# --- Tool Input Schema ---
class CustomerServiceInput(BaseModel):
    request_type: ServiceType = Field(description="The specific category of the service being requested (e.g., ROOM_SERVICE, DINING_BOOKING).")
    details: str = Field(description="The specific details of the user's request (e.g., 'extra towels', 'table for 2 at 7 PM', 'couples massage tomorrow').")
    phone_number: str = Field(description="The user's phone number, required for context and processing the request.")
    preferences_summary: Optional[str] = Field(None, description="A summary of the user's known preferences.") # Optional, but useful

# --- The Tool ---
class CustomerServiceTool(BaseTool):
    """
    Tool to handle specific guest service requests like Room Service, Dining Reservations, Spa Bookings, and Cabana inquiries.
    It attempts to forward the request to the relevant department via email (or simulates it) for certain types,
    and provides specific instructions for others (like Cabana bookings).
    """
    name: str = "handle_customer_service_request"
    description: str = (
        "Use this tool ONLY for specific guest service requests like Room Service, Dining Reservations, Spa Bookings, or Cabana inquiries. "
        "**IMPORTANT:** Before calling this tool for DINING_BOOKING, you MUST have the desired date, time, and number of guests. "
        "Before calling for CABANA_BOOKING, you MUST have the desired date. "
        "Before calling for SPA_BOOKING, you MUST have the desired date and preferred time/service type. "
        "If these details are missing from the user's request, ask clarifying questions first. "
        "Once ALL required details are gathered, call this tool. "
        "Provide the appropriate 'request_type' (ROOM_SERVICE, DINING_BOOKING, SPA_BOOKING, CABANA_BOOKING) and the FULL 'details' gathered."
    )
    args_schema: Type[BaseModel] = CustomerServiceInput

    # --- SMTP Configuration (Copied from Controller) ---
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    recipient_email: Optional[str] = None
    smtp_configured: bool = False
    # --- Special Handling Instructions ---
    # Define which services use email vs. giving instructions
    EMAIL_SERVICE_TYPES = {ServiceType.ROOM_SERVICE, ServiceType.DINING_BOOKING, ServiceType.SPA_BOOKING}
    INSTRUCTION_SERVICE_TYPES = {ServiceType.CABANA_BOOKING: "For Cabana bookings, please call our Pool Concierge directly at 1-800-CABANAS (example number)."}

    def __init__(self, **kwargs):
        super().__init__(**kwargs) # Pass any BaseTool args
        # Load SMTP config from environment variables
        self.smtp_server = os.getenv("SMTP_SERVER")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.recipient_email = os.getenv("CUSTOMER_SERVICE_EMAIL")

        self.smtp_configured = all([self.smtp_server, self.smtp_port, self.smtp_user, self.smtp_password, self.recipient_email])

        if not self.smtp_configured:
            logger.warning("CustomerServiceTool: SMTP configuration is incomplete. Email notifications will be disabled (or stubbed if STUB_EMAIL_SENDING is True).")
        else:
            logger.info(f"CustomerServiceTool initialized. Emails configured for {self.recipient_email}. STUB_EMAIL_SENDING={STUB_EMAIL_SENDING}")

    def _send_request_email(self, request_type: str, message_details: str, user_phone: str, prefs_summary: Optional[str]) -> bool:
        """Internal method to send email or stub it."""
        # --- START Stubbing Logic ---
        if STUB_EMAIL_SENDING:
            logger.info(f"--- STUBBED EMAIL (Simulating Success) via CustomerServiceTool ---")
            logger.info(f"Recipient: {self.recipient_email}")
            logger.info(f"Request Type: {request_type}")
            logger.info(f"From Phone: {user_phone}")
            logger.info(f"Preferences: {prefs_summary if prefs_summary else 'None Recorded'}")
            logger.info(f"Request Details: {message_details}")
            logger.info("Request sent to guest services Successfully (Simulated)")
            return True
        # --- END Stubbing Logic ---

        if not self.smtp_configured:
            logger.error("SMTP is not configured. Cannot send email.")
            return False

        # Build and send email (same logic as before)
        subject = f"New {request_type.replace('_', ' ').title()} Request from {user_phone}"
        email_content = f"""
        Guest Phone: {user_phone}
        Request Type: {request_type}

        Known Preferences:
        --------------------
        {prefs_summary if prefs_summary else 'None Recorded'}
        --------------------

        Request Details:
        --------------------
        {message_details}
        --------------------

        Please process this request.
        """
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
            # Simplified error logging for brevity, keep detailed if needed
            logger.error(f"Failed to send SMTP email for {request_type} from {user_phone}: {e}", exc_info=True)
            return False

    def _generate_fulfillment_response(self, request_type: ServiceType) -> str:
        """Generates standard confirmation for email-based requests."""
        responses = {
            ServiceType.ROOM_SERVICE: "Okay, I've passed your room service request ({details}) to our team.^MSG^They'll process it shortly.",
            ServiceType.DINING_BOOKING: "Thanks! I've forwarded your dining reservation request ({details}).^MSG^Someone will confirm availability with you if needed.",
            ServiceType.SPA_BOOKING: "Got it. I've sent your spa booking request ({details}) to the wellness center.^MSG^They'll reach out to confirm details.",
        }
        # Default added for safety, though should be covered by EMAIL_SERVICE_TYPES check
        return responses.get(request_type, "Okay, I'm processing your request ({details}).")

    def _generate_cannot_fulfill_response(self, request_type: ServiceType, reason: str = "system issue") -> str:
        """Generates message indicating request cannot be fulfilled automatically."""
        if reason == "smtp_disabled":
             issue = "our notification system is currently unavailable"
        elif reason == "smtp_failed":
             issue = "there was a problem sending your request to the team"
        else: # General catch-all
             issue = "I encountered an issue processing that automatically"

        base_message = f"I understand you'd like help with {request_type.value.replace('_', ' ').lower()} ({details}). However, {issue}."
        suggestion = "Could you please call the front desk or concierge directly for assistance?"
        return f"{base_message}^MSG^{suggestion}"

    def _run(self, request_type: ServiceType, details: str, phone_number: str, preferences_summary: Optional[str] = None) -> str:
        """Executes the tool's logic based on the request type."""
        logger.info(f"CustomerServiceTool invoked: type='{request_type}', details='{details}', phone='{phone_number}'")

        # --- Handle Email-based Services ---
        if request_type in self.EMAIL_SERVICE_TYPES:
            if not self.smtp_configured and not STUB_EMAIL_SENDING:
                logger.warning(f"Cannot fulfill {request_type} for {phone_number} because SMTP is not configured and stubbing is OFF.")
                return self._generate_cannot_fulfill_response(request_type, reason="smtp_disabled").format(details=details)

            email_sent_or_simulated = self._send_request_email(request_type.value, details, phone_number, preferences_summary)

            if email_sent_or_simulated:
                return self._generate_fulfillment_response(request_type).format(details=details)
            else:
                # Only reached if STUBBING is False and send failed
                return self._generate_cannot_fulfill_response(request_type, reason="smtp_failed").format(details=details)

        # --- Handle Instruction-based Services ---
        elif request_type in self.INSTRUCTION_SERVICE_TYPES:
            instruction = self.INSTRUCTION_SERVICE_TYPES[request_type]
            logger.info(f"Providing instructions for {request_type} for {phone_number}.")
            # Return the specific instruction defined for this type
            return f"Okay, regarding your request for {request_type.value.replace('_', ' ').lower()} ({details}):^MSG^{instruction}"

        # --- Handle Unknown/Unsupported Service Types ---
        else:
            # This case *shouldn't* happen if the agent uses the Enum correctly based on the description
            logger.warning(f"CustomerServiceTool received unsupported request_type: {request_type}")
            return f"I understand you're asking about {details}, but I'm not equipped to handle '{request_type.value}' requests directly. Please contact the front desk."

    async def _arun(self, request_type: ServiceType, details: str, phone_number: str, preferences_summary: Optional[str] = None) -> str:
        """Async version simply calls the sync version for now."""
        # If _send_request_email becomes async, this needs proper async implementation
        return self._run(request_type, details, phone_number, preferences_summary)