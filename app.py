# my_project/app.py

import os
import time
import logging
import json
import datetime

from dotenv import load_dotenv
from flask import Flask, request
from twilio.rest import Client

# --- Langchain Imports ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    import faiss
except ImportError:
    logging.error("FAISS library not found. Please install it: pip install faiss-cpu")
    faiss = None

# --- Local Module Imports ---
from models.file_storage_helper import FileStorageHelper
from models.spam_filter import SpamFilter
from controllers.weather_call import WeatherController
from controllers.tavily import research as tavily_research
from controllers.google_map import GooglePlacesTool
from controllers.profiler_agent import ProfilerAgent

# ---> Import ALL Agent Tools from agent_tools.py <---
from agent_tools import (
    GetCurrentTimeTool,
    FacilitiesSearchTool,
    PreferencesSearchTool,
    BestPracticesSearchTool,
    WeatherTool,
    TavilySearchTool,
    CustomerServiceTool # Import the new tool from its consolidated location
)

# --- Import Orchestrator-used Tools (Keep as is) ---
from tools import (
    PreferenceLoaderTool,
    SpamFilterTool,
    MessageSenderTool
)

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---


# 1) Load environment
load_dotenv(override=True)
# Ensure all required env vars are checked (OpenAI, Twilio, Google Maps, SMTP for CustomerServiceTool)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
SMTP_SERVER = os.getenv("SMTP_SERVER")
CUSTOMER_SERVICE_EMAIL = os.getenv("CUSTOMER_SERVICE_EMAIL")
# Add checks if needed
if not all([OPENAI_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER]):
    logger.critical("Core API keys (OpenAI, Twilio) are missing. Application cannot function.")
    # Consider exiting or raising an error in a real application
if not all([SMTP_SERVER, CUSTOMER_SERVICE_EMAIL, os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD")]):
     logger.warning("SMTP configuration is incomplete in .env. Customer Service email feature might be disabled or fail if STUB_EMAIL_SENDING=False.")

# --- Flask App and Twilio Client ---
app = Flask(__name__)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
# --- End Flask App and Twilio Client ---


# 2) Helper functions for preferences (Keep as is)
file_storage_helper = FileStorageHelper()
preferences_store_map = {}

def get_prefs_filename(phone_number: str) -> str:
    safe_number = "".join(ch for ch in phone_number if ch.isalnum() or ch in ['+', '_'])
    return f"customer_{safe_number}_prefs.txt"

# Function get_user_preferences_store (Keep existing implementation)
def get_user_preferences_store(phone_number: str):
    """Gets or creates a FAISS vector store for user preferences."""
    if phone_number in preferences_store_map:
        if preferences_store_map[phone_number] is not None:
            logger.debug(f"Returning cached preference store for {phone_number}")
            return preferences_store_map[phone_number]
        else:
            logger.warning(f"Cached store for {phone_number} was None. Rebuilding.")

    filename = get_prefs_filename(phone_number)
    prefs_texts = file_storage_helper.load_lines(filename)
    store = None

    if prefs_texts:
        valid_prefs_texts = [text for text in prefs_texts if text and text.strip()] # Ensure non-empty strings
        if valid_prefs_texts:
            try:
                logger.info(f"Loading preference store for {phone_number} from {len(valid_prefs_texts)} lines.")
                if embeddings_model:
                    store = FAISS.from_texts(valid_prefs_texts, embeddings_model)
                else:
                    logger.error("Embeddings model not initialized. Cannot create FAISS store.")
            except Exception as e:
                logger.error(f"Could not load FAISS from texts for {phone_number}: {e}", exc_info=True)
                store = None
        else:
            logger.info(f"Preference file {filename} for {phone_number} is empty or contains only whitespace.")

    if store is None:
        logger.info(f"Creating new empty FAISS store for {phone_number} (File: {filename}).")
        if faiss and embeddings_model:
            try:
                # Ensure embedding model is ready, e.g., by embedding a test query
                test_embedding = embeddings_model.embed_query("test")
                embedding_dimension = len(test_embedding)
                index = faiss.IndexFlatL2(embedding_dimension)
                docstore = InMemoryDocstore({}) # Use default empty docstore
                index_to_docstore_id = {} # Use default empty mapping
                store = FAISS(
                    embedding_function=embeddings_model.embed_documents, # Pass the function directly
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id
                )
                logger.info(f"Successfully created empty FAISS store for {phone_number}.")
            except Exception as e:
                logger.error(f"Error creating empty FAISS store for {phone_number}: {e}", exc_info=True)
                store = None
        else:
             if not faiss: logger.error("FAISS library unavailable.")
             if not embeddings_model: logger.error("Embeddings model unavailable.")
             store = None

    preferences_store_map[phone_number] = store
    return store


# --- Initialize Core Components ---
#llm_model = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.2)
llm_model = ChatOpenAI(model_name="gpt-4.1-nano", openai_api_key=OPENAI_API_KEY, temperature=0.75)
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
spam_filter = SpamFilter(embeddings_model)
weather_controller = WeatherController()
# REMOVE service matcher/controller instantiation
# file_storage_helper already instantiated

# --- Load Domain Data (Keep existing logic) ---
logger.info("Loading and splitting domain data...")
facilities_vector_store = None
best_practices_vector_store = None
# (Keep the try-except blocks for loading facilities.txt and best_practices.txt)
try:
    facilities_lines = file_storage_helper.load_lines("facilities.txt")
    if facilities_lines:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        facility_documents = text_splitter.create_documents(["\n".join(facilities_lines)])
        if facility_documents: facilities_vector_store = FAISS.from_documents(facility_documents, embeddings_model)
    if facilities_vector_store: logger.info("Facilities data loaded.")
    else: logger.warning("Facilities data could not be loaded into vector store.")
except Exception as e: logger.exception(f"Error loading facilities data: {e}")

try:
    bp_lines = file_storage_helper.load_lines("best_practices.txt")
    if bp_lines:
        text_splitter_bp = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        bp_documents = text_splitter_bp.create_documents(["\n".join(bp_lines)])
        if bp_documents: best_practices_vector_store = FAISS.from_documents(bp_documents, embeddings_model)
    if best_practices_vector_store: logger.info("Best practices data loaded.")
    else: logger.warning("Best practices data could not be loaded into vector store.")
except Exception as e: logger.exception(f"Error loading best practices data: {e}")
# --- End Load Domain Data ---


# --- Instantiate Profiler Agent (Keep as is) ---
logger.info("Initializing Profiler Agent...")
profiler_agent = ProfilerAgent(
    llm=llm_model,
    get_prefs_filename=get_prefs_filename,
    file_storage_helper=file_storage_helper,
    get_user_preferences_store=get_user_preferences_store
)
logger.info("Profiler Agent initialized.")
# ---------------------------------


# --- Instantiate Tools ---
logger.info("Initializing tools...")

# ---> Instantiate ALL agent tools <---
# Note: Pass dependencies like vector stores and controllers here
agent_tool_instances = [
    GetCurrentTimeTool(),
    PreferencesSearchTool(get_user_preferences_store=get_user_preferences_store), # Pass the function
    WeatherTool(weather_controller=weather_controller), # Pass the controller instance
    TavilySearchTool(),
    CustomerServiceTool() # Instantiate the service tool
]
# Conditionally add tools based on loaded data/config
if facilities_vector_store:
    agent_tool_instances.append(FacilitiesSearchTool(vector_store=facilities_vector_store))
    logger.info("FacilitiesSearchTool added.")
else: logger.warning("Facilities vector store unavailable. FacilitiesSearchTool disabled.")

if best_practices_vector_store:
    agent_tool_instances.append(BestPracticesSearchTool(vector_store=best_practices_vector_store))
    logger.info("BestPracticesSearchTool added.")
else: logger.warning("Best practices vector store unavailable. BestPracticesSearchTool disabled.")

if GOOGLE_MAPS_API_KEY:
    try:
        google_places_tool = GooglePlacesTool(api_key=GOOGLE_MAPS_API_KEY)
        agent_tool_instances.append(google_places_tool)
        logger.info("Google Places Tool added.")
    except Exception as e: logger.error(f"Failed to initialize Google Places Tool: {e}")
else: logger.warning("GOOGLE_MAPS_API_KEY not set. Google Places Tool disabled.")


# App-level tools (Keep as is)
spam_filter_tool = SpamFilterTool(spam_filter)
preference_loader_tool = PreferenceLoaderTool(
    file_storage_helper=file_storage_helper,
    get_prefs_filename=get_prefs_filename
)
message_sender_tool = MessageSenderTool(
    twilio_client=twilio_client,
    from_number=TWILIO_FROM_NUMBER
)
logger.info(f"Total agent tools initialized: {len(agent_tool_instances)}")
# --- End Instantiate Tools ---


# --- Create Agent (Using Updated Prompt for Strategy 1) ---
logger.info("Creating Agent...")
# ---> Updated Prompt for Strategy 1 (Agent asks clarifying questions) <---
AGENT_SYSTEM_PROMPT = """
You are Ryly, a butler at a Planet Hollywood Resort in Cancun. Read the [Definitions] and [Guidelines] and follow the [steps] and [Instructions].

            [Definitions]
             [Instructions] Guidelines as to how you should response
             [Steps] Steps that you will follow in your analysis and execution
             [Relevant Info] Information about the resort hotel facility, weather, web search results etc.
             [Conversation-Flow] Responding framework
             [Guidelines] Instructions on how to handle certain topics
             [/Definitions]

             [Steps]
             [Step-1]Analyze the user's query and the `chat_history` for intent. Select a [Conversation-Flow] that matches the intend[/Step-1]
             [Step-3]If you cannot find an answer or fulfill the request after following the steps, respond politely indicating you cannot help with that specific part right now.[/step-3]
             [/Steps]

             [Guidelines]
             [Guideline-1]Weather for Recommendations: Consider the time and weather for activity/excursion suggestions.[/Guideline-1]
             [Guideline-2]When helping with activities and facilities inquiry: Consider time of the day, weather, preferences. Suggest relevant activities.[/Guideline-2]
             [Guideline-3]Location Context: 'here' means Planet Hollywood, Cancun.[/Guideline-3]
             [Guideline-4]Internal Info: Check `search_resort_facilities` first for resort food/activities.[/Guideline-3]
             [Guideline-5]External Info: Use `web_search` for general Cancun info or things outside the resort.[/Guideline-3]
             [/Guidelines]

             [Conversation-Flow]
             [Ask Clarifying Question(s)]
             If you can’t fulfill the request because information is missing (especially for Service Requests per [Guideline-1]), ask specific clarifying question(s).
             Use short, dependency-focused sentence.
             [/Ask Clarifying Question(s)]

             [Offer Recommendation then Ask]
             If the user’s request suggests multiple or personalized options, propose the best fit first.
             Then, confirm alignment with a quick question (e.g., “Does that match what you’re looking for?”).
             [/Offer Recommendation then Ask]

             [Answer Directly]
             If the request is straightforward and you have a single clear answer (or after a tool like `handle_customer_service_request` provides the answer), respond immediately.
             Use short dependency-linked phrases to keep the flow precise.
             [/Answer Directly]

             [Service Request]
             If Room Service Request (eg. need towels, need shampoo, need amendities, need water and etc) - Ensure the guest provide their needs.
             If Cabana Booking and only the action of making a reservation - You must get the date and time; if not ask for it.
             If SPA Booking and only the action of making a reservation - Say you cannot provide this service at this time, and have them reach out to Guest Services
             For Room Services, Dining Booking and Cabana Booking, **Crucially:** Once `handle_customer_service_request` returns its result (confirmation or instructions), that result IS the final response for this service request. **Do not attempt other tool calls or actions for this specific request.
             [/Service Request]

             [Resturant Booking]
             If Dining Booking and only the action of making a reservation - You must get the date, time and number of guests; if not ask for it.
             [/ResturantBooking]

             [Information Request]
             [Step-1]Determine the user's need (recommendation, facility info, weather, external search etc).[/Step-1]
             [Step-2]Select the appropriate tool(s) based on [Guidelines] and tool descriptions (e.g., `search_resort_facilities`, `get_cancun_weather`, `web_search`).[/Step-2]
             [Step-3]Invoke the necessary tool(s).[/Step-3]
             [Step-4]Synthesize the results from the tool(s), `chat_history`, and `preferences_summary` into a concise, helpful response using the [Instructions] and [Conversation-Flow].[/Step-4]
             [/Information Request]
             [/Conversation-Flow]

             [Instructions]
             [instruction-1]Maintain a conversational and joyful yet professional tone.[/instruction-1]
             [instruction-2]Use the dependency grammar linguistic framework rather than phrase structure grammar for the output. The idea is that the closer together each pair of words you’re connecting are, the easier the copy will be to comprehend.[/instruction-2]
             [instruction-3]Break messages naturally, around 2–3 sentences per bubble.[/instruction-3]
             [instruction-4]Use the delimiter ^MSG^ between bubbles.[/instruction-4]
             [instruction-5]Provide helpful and concise responses without unnecessary commentaries.[/instruction-5]
             [/Instructions]
"""
# ---> End Prompt Update <---
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        # Pass phone_number and preferences_summary in the user message context
        ("user", "Guest's phone number: {phone_number}\nKnown Preferences Summary:\n{preferences_summary}\n\nUser Query: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
# Pass the list of tool *instances*
agent = create_openai_tools_agent(llm_model, agent_tool_instances, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=agent_tool_instances, verbose=True, handle_parsing_errors="Check the output and correct the parsing error." # More informative error handling
)
logger.info("Agent created.")
# --- End Create Agent ---


# --- Conversation Tracking & Memory (Keep as is) ---
conversation_histories = {}
def get_agent_memory(history_str: str):
    """Creates ConversationBufferMemory from a simple string history."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    lines = history_str.strip().split('\n')
    for i in range(0, len(lines), 2):
        user_line = lines[i]
        ai_line = lines[i+1] if (i+1) < len(lines) else ""
        # Basic parsing, assumes "User: " and "AI: " prefixes might exist but handles if not
        user_msg = user_line.split(":", 1)[1].strip() if ":" in user_line else user_line.strip()
        ai_msg = ai_line.split(":", 1)[1].strip() if ":" in ai_line else ai_line.strip()
        if user_msg: memory.chat_memory.add_user_message(user_msg)
        if ai_msg: memory.chat_memory.add_ai_message(ai_msg)
    return memory


# --- Flask Route ---
@app.route("/demo-reply", methods=['POST'])
def incoming_message():
    start_time = time.time()
    incoming_msg = request.form.get('Body', '').strip()
    from_number = request.form.get('From', '')
    logger.info(f"Received message from {from_number}: '{incoming_msg}'")

    if not incoming_msg:
        logger.warning(f"Received empty message from {from_number}. Ignoring.")
        return "<Response></Response>"

    # 1) Get/Init History & Pref Store
    if from_number not in conversation_histories:
        logger.info(f"New conversation started for {from_number}")
        conversation_histories[from_number] = ""
    # Ensure pref store is initialized/retrieved (robustness)
    try:
        prefs_store = get_user_preferences_store(from_number) # Call ensures it's loaded/created
        if prefs_store is None:
            logger.error(f"Preference store is None for {from_number} after get/init attempt.")
    except Exception as e:
        logger.exception(f"Error ensuring preference store exists for {from_number}: {e}")


    user_history_str = conversation_histories.get(from_number, "")

    # 2) Spam Check
    if spam_filter_tool.run(incoming_msg):
        logger.warning(f"Spam detected from {from_number}: '{incoming_msg}'")
        deny_msg = "I’m sorry, but I can’t help with that request."
        message_sender_tool.run(to_number=from_number, message_body=deny_msg)
        return "<Response></Response>"

    # 3) Load Prefs Summary (Needed for Agent Input)
    prefs_summary = "No preferences recorded yet." # Default
    try:
        # This still relies on PreferenceLoaderTool reading the .txt file
        prefs_summary_loaded = preference_loader_tool.run(from_number)
        if prefs_summary_loaded and prefs_summary_loaded.strip() and prefs_summary_loaded.strip().lower() != 'none':
            prefs_summary = prefs_summary_loaded.strip()
        else:
             logger.info(f"No preference summary loaded for {from_number} (or file content is 'None').")
    except Exception as e:
        logger.error(f"Error loading preference summary for {from_number}: {e}")

    # ---> 4) REMOVED Customer Service Pre-Check <---
    # Agent handles service requests now

    # ---> 5) Proceed DIRECTLY with Agent <---
    logger.info(f"Proceeding with Agent for message from {from_number}.")
    agent_memory = get_agent_memory(user_history_str)
    response_text = "Sorry, I had trouble processing that." # Default agent error response

    try:
        logger.info(f"Invoking agent executor for {from_number}...")
        agent_input = {
            "input": incoming_msg,
            "chat_history": agent_memory.chat_memory.messages,
            "phone_number": from_number, # Crucial for PrefsTool and ServiceTool
            "preferences_summary": prefs_summary # Provide context to agent/tools
        }
        loggable_input = {k: v if k != "chat_history" else "[Messages Redacted]" for k, v in agent_input.items()}
        logger.debug(f"Agent Input: {json.dumps(loggable_input, default=str)}")

        # Invoke the agent
        response = agent_executor.invoke(agent_input)
        agent_output = response.get("output")

        if agent_output and agent_output.strip():
            response_text = agent_output.strip()
            logger.info(f"Agent generated response for {from_number}.")
        else:
             logger.warning(f"Agent for {from_number} returned empty/None output. Raw Response: {response}")
             response_text = "I'm sorry, I couldn't generate a response for that right now."

    except Exception as e:
        # Log the full exception traceback
        logger.exception(f"Critical error during agent execution for {from_number}: {e}")
        # Provide a user-friendly error message
        response_text = "I encountered an unexpected problem processing your request. Please try again shortly or contact the front desk if the issue persists."

    # ---> 6) Update History (Agent conversation) <---
    # Store the AI response (can combine multi-bubble for logging simplicity)
    stored_ai_response = response_text.replace('^MSG^', ' ')
    # Format history string consistently
    updated_history = user_history_str + f"User: {incoming_msg}\nAI: {stored_ai_response}\n"
    conversation_histories[from_number] = updated_history
    logger.debug(f"History updated for {from_number}.")

    # ---> 7) Finalize Preferences using ProfilerAgent (After agent interaction) <---
    # Run this *after* the agent interaction, using the *updated* history
    try:
        logger.info(f"Running preference profiling for {from_number}...")
        profiling_result = profiler_agent.profile_user(
            phone_number=from_number,
            conversation_history=updated_history # Use the history including the latest turn
        )
        logger.info(f"Preference profiling result for {from_number}: {profiling_result}")
    except Exception as e:
        logger.exception(f"Error during preference profiling for {from_number}: {e}")
        # Don't overwrite agent response if profiling fails

    # ---> 8) Send Agent Response <---
    try:
        logger.info(f"Sending agent response to {from_number}...")
        # Send the original agent output, preserving ^MSG^ for multi-bubble messages
        message_sender_tool.run(to_number=from_number, message_body=response_text)
        logger.info(f"Agent response sent to {from_number}.")
    except Exception as e:
        logger.exception(f"Error sending agent message via Twilio to {from_number}: {e}")
        # Potentially notify monitoring or try fallback

    end_time = time.time()
    logger.info(f"Request from {from_number} processed in {end_time - start_time:.2f} seconds.")
    return "<Response></Response>"

# --- Main Execution ---
if __name__ == "__main__":
    if not faiss:
        logger.critical("FAISS library not found or failed to import. Vector store functionality will be severely limited.")
    # Check agent tools initialized
    if not agent_tool_instances:
        logger.critical("No agent tools were successfully initialized. Agent functionality will be minimal.")
    else:
        logger.info(f"Agent ready with {len(agent_tool_instances)} tools.")

    logger.info("Starting Flask application...")
    # Use host="0.0.0.0" for Docker/network access
    # Set debug=False for production environments
    app.run(host="0.0.0.0", port=55555, debug=True) # Keep debug=True for development visibility