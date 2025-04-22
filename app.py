# my_project/app.py

import os
import time
import logging
import json
import datetime # <-- Import datetime for history timestamps later if needed

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

# --- FAISS Direct Import ---
try:
    import faiss
except ImportError:
    logging.error("FAISS library not found. Please install it: pip install faiss-cpu")
    faiss = None

# Our local modules
from models.file_storage_helper import FileStorageHelper
from models.spam_filter import SpamFilter
# ---> Import the new Matcher and Controller <---
from models.service_matcher import CustomerServiceMatcher
from controllers.customer_service import CustomerServiceController
# ---> End Imports <---
from controllers.weather_call import WeatherController
from controllers.tavily import research as tavily_research
from controllers.google_map import GooglePlacesTool
from controllers.profiler_agent import ProfilerAgent

# Import New Granular Agent Tools
from agent_tools import (
    GetCurrentTimeTool,
    FacilitiesSearchTool,
    PreferencesSearchTool,
    BestPracticesSearchTool,
    WeatherTool,
    TavilySearchTool
)

# Import Orchestrator-used Tools
from tools import (
    PreferenceLoaderTool,
    # PreferenceUpdateTool, # Currently likely incompatible due to JSON/TXT mismatch
    SpamFilterTool,
    MessageSenderTool
)

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---


# 1) Load environment
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
# ---> Add SMTP env vars check (optional but good practice) <---
SMTP_SERVER = os.getenv("SMTP_SERVER")
CUSTOMER_SERVICE_EMAIL = os.getenv("CUSTOMER_SERVICE_EMAIL")
if not SMTP_SERVER or not CUSTOMER_SERVICE_EMAIL:
    logger.warning("SMTP_SERVER or CUSTOMER_SERVICE_EMAIL not set in .env. Customer service request handling might be limited.")
# --- End Load environment ---


# --- Flask App and Twilio Client ---
app = Flask(__name__)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
# --- End Flask App and Twilio Client ---


# 2) Helper functions for preferences (Keep as is)
file_storage_helper = FileStorageHelper()
preferences_store_map = {}

def get_prefs_filename(phone_number: str) -> str:
    safe_number = "".join(ch for ch in phone_number if ch.isalnum() or ch in ['+', '_'])
    return f"customer_{safe_number}_prefs.txt" # Keeping .txt for now

# --- Initialize Core Components ---
llm_model = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.2)
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
spam_filter = SpamFilter(embeddings_model)
weather_controller = WeatherController()
# ---> Instantiate Matcher and Controller <---
customer_service_matcher = CustomerServiceMatcher(embeddings_model=embeddings_model, file_path="customer_services.txt")
customer_service_controller = CustomerServiceController()
# ---> End Instantiation <---
# file_storage_helper already instantiated above

# Function get_user_preferences_store (Keep as is)
def get_user_preferences_store(phone_number: str):
    """
    Gets or creates a FAISS vector store for user preferences.
    Loads from lines in a .txt file or creates an empty store if needed.
    Uses direct FAISS creation instead of deprecated helper.
    """
    if phone_number in preferences_store_map:
        if preferences_store_map[phone_number] is not None:
            logger.debug(f"Returning cached preference store for {phone_number}")
            return preferences_store_map[phone_number]
        else:
            logger.warning(f"Cached store for {phone_number} was None. Attempting to rebuild.")

    filename = get_prefs_filename(phone_number)
    prefs_texts = file_storage_helper.load_lines(filename)
    store = None

    if prefs_texts:
        valid_prefs_texts = [text for text in prefs_texts if text.strip()]
        if valid_prefs_texts:
            try:
                logger.info(f"Attempting to load preference store for {phone_number} from text file: {filename}")
                if embeddings_model:
                    store = FAISS.from_texts(valid_prefs_texts, embeddings_model)
                    logger.info(f"Loaded FAISS store from {len(valid_prefs_texts)} lines for {phone_number}.")
                else:
                    logger.error("Embeddings model not initialized. Cannot create FAISS store from text.")
            except Exception as e:
                logger.error(f"Could not load FAISS from texts for {phone_number} (file: {filename}, error: {e}). Will create empty store.", exc_info=True)
                store = None
        else:
            logger.info(f"Preference file {filename} for {phone_number} contains only empty lines. Creating empty store.")
            store = None

    if store is None:
        logger.info(f"Creating new empty FAISS store for {phone_number} (File: {filename}).")
        if faiss and embeddings_model:
            try:
                embedding_dimension = len(embeddings_model.embed_query("test"))
                index = faiss.IndexFlatL2(embedding_dimension)
                docstore = InMemoryDocstore({})
                store = FAISS(
                    embedding_function=embeddings_model,
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id={}
                )
                logger.info(f"Successfully created empty FAISS store for {phone_number}.")
            except Exception as e:
                logger.error(f"An unexpected error occurred creating empty FAISS store for {phone_number}: {e}", exc_info=True)
                store = None
        else:
             if not faiss: logger.error("FAISS library not installed or failed to import. Cannot create empty store.")
             if not embeddings_model: logger.error("Embeddings model not initialized. Cannot create empty store.")
             store = None

    preferences_store_map[phone_number] = store
    return store


# --- Load Domain Data (Keep as is) ---
logger.info("Loading and splitting facilities data...")
facilities_vector_store = None
try:
    facilities_lines = file_storage_helper.load_lines("facilities.txt")
    facilities_full_text = "\n".join(facilities_lines)
    if facilities_full_text.strip():
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        facility_documents = text_splitter.create_documents([facilities_full_text])
        if facility_documents:
            facilities_vector_store = FAISS.from_documents(facility_documents, embeddings_model)
            logger.info(f"Facilities data loaded and split into {len(facility_documents)} documents.")
        else: logger.warning("Text splitting of facilities.txt resulted in zero documents.")
    else: logger.error("facilities.txt is empty or could not be read.")
except Exception as e: logger.exception(f"Error loading or splitting facilities.txt: {e}")

logger.info("Loading and splitting best practices data...")
best_practices_vector_store = None
try:
    best_practices_lines = file_storage_helper.load_lines("best_practices.txt")
    best_practices_full_text = "\n".join(best_practices_lines)
    if best_practices_full_text.strip():
        text_splitter_bp = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        bp_documents = text_splitter_bp.create_documents([best_practices_full_text])
        if bp_documents:
            best_practices_vector_store = FAISS.from_documents(bp_documents, embeddings_model)
            logger.info(f"Best practices data loaded and split into {len(bp_documents)} documents.")
        else: logger.warning("Text splitting of best_practices.txt resulted in zero documents.")
    else: logger.warning("best_practices.txt is empty or could not be read.")
except Exception as e: logger.exception(f"Error loading or splitting best_practices.txt: {e}")
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


# --- Instantiate Tools (Keep Agent tools as is) ---
logger.info("Initializing tools...")
agent_tools = [
    GetCurrentTimeTool(),
    PreferencesSearchTool(get_user_preferences_store=get_user_preferences_store),
    WeatherTool(weather_controller=weather_controller),
    TavilySearchTool()
]
# ... (rest of agent_tools setup remains the same)
if facilities_vector_store:
    agent_tools.append(FacilitiesSearchTool(vector_store=facilities_vector_store))
    logger.info("FacilitiesSearchTool initialized.")
else: logger.warning("Facilities vector store not available. FacilitiesSearchTool will be disabled.")
if best_practices_vector_store:
    agent_tools.append(BestPracticesSearchTool(vector_store=best_practices_vector_store))
    logger.info("BestPracticesSearchTool initialized.")
else: logger.warning("Best practices vector store not available. BestPracticesSearchTool will be disabled.")
if GOOGLE_MAPS_API_KEY:
    try:
        google_places_tool = GooglePlacesTool(api_key=GOOGLE_MAPS_API_KEY)
        agent_tools.append(google_places_tool)
        logger.info("Google Places Tool initialized.")
    except Exception as e: logger.error(f"Failed to initialize Google Places Tool: {e}. It will not be available.")
else: logger.warning("GOOGLE_MAPS_API_KEY not set. Google Places Tool will not be available.")

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
logger.info("Tools initialized.")
# --- End Instantiate Tools ---


# --- Create Agent (Keep as is) ---
logger.info("Creating Agent...")
AGENT_SYSTEM_PROMPT = """
You are Ryly, a helpful and joyful digital butler at the **Planet Hollywood Cancun** luxury resort.
Your goal is to assist guests with their requests accurately and efficiently.
You have access to several tools to gather information. Use them ONLY when necessary based on the user's query.
**When the guest asks about things 'here', 'nearby', 'around here', or 'at the resort', assume they mean Planet Hollywood Cancun unless they specify a different location.**

Thinking Process:
1. Analyze the user's query. Do they mean inside Planet Hollywood Cancun, the wider Cancun area, or somewhere else? Check the `chat_history` for context.
2. If the query is about 'here' or the resort, check `search_resort_facilities` first.
3. Determine which other tool(s), if any, are REQUIRED based on the guidelines.
4. Call the necessary tools.
5. Stop calling tools once you have enough information.
6. Synthesize the information gathered, user preferences (`preferences_summary`), and history (`chat_history`) into a helpful, concise final answer, remembering you represent Planet Hollywood Cancun.
7. **Recommendation Nuance:** When providing recommendations (like restaurants), if the user asks for alternatives or refines their criteria (e.g., asks for 'family-friendly' after you suggested a specific place), **offer *different* options** that match the new criteria based on tool results and history. **Avoid repeating the immediately preceding suggestion** unless the user explicitly asks about it again or confirms it's okay.

Response Formatting:
- Respond conversationally, professionally, and concisely.
- Briefly incorporate key tool results.
- If info isn't found, politely state that.
- Format the final response with '^MSG^' separating bubbles (2-4 sentences per bubble).
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Guest's phone number: {phone_number}\nKnown Preferences Summary:\n{preferences_summary}\n\nUser Query: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
agent = create_openai_tools_agent(llm_model, agent_tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=agent_tools, verbose=True, handle_parsing_errors=True
)
logger.info("Agent created.")
# --- End Create Agent ---


# --- Conversation Tracking & Memory (Keep as is) ---
conversation_histories = {}
def get_agent_memory(history_str: str):
    # Keep timestamp stripping for now if history doesn't have them consistently
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    lines = history_str.strip().split('\n')
    for i in range(0, len(lines), 2):
        user_line = lines[i]
        ai_line = lines[i+1] if (i+1) < len(lines) else ""

        # Simple split logic (adjust if timestamps ARE present)
        user_msg = user_line[len("User: "):] if user_line.startswith("User: ") else user_line
        ai_msg = ai_line[len("AI: "):] if ai_line.startswith("AI: ") else ai_line

        if user_msg: memory.chat_memory.add_user_message(user_msg)
        if ai_msg: memory.chat_memory.add_ai_message(ai_msg)
    return memory

# --- Flask Route ---
@app.route("/demo-reply", methods=['POST'])
def incoming_message():
    incoming_msg = request.form.get('Body', '').strip() # Added strip()
    from_number = request.form.get('From', '')
    logger.info(f"Received message from {from_number}: '{incoming_msg}'")

    if not incoming_msg:
        logger.warning(f"Received empty message from {from_number}. Ignoring.")
        return "<Response></Response>" # Ignore empty messages

    # 1) Get/Init History & Pref Store (Needed early for both paths)
    if from_number not in conversation_histories:
        logger.info(f"New conversation started for {from_number}")
        conversation_histories[from_number] = ""
        prefs_store = get_user_preferences_store(from_number)
        if prefs_store is None:
            logger.error(f"Failed to initialize preference store for {from_number}. Preference features may be limited.")
    else:
        prefs_store = get_user_preferences_store(from_number)
        if prefs_store is None:
             logger.error(f"Failed to retrieve or rebuild preference store for {from_number}. Preference features may be limited.")

    user_history_str = conversation_histories.get(from_number, "")

    # 2) Spam Check
    if spam_filter_tool.run(incoming_msg):
        logger.warning(f"Spam detected from {from_number}: '{incoming_msg}'")
        deny_msg = "I’m sorry, but I can’t help with that request."
        # ---> Directly use MessageSenderTool instance <---
        message_sender_tool.run(to_number=from_number, message_body=deny_msg)
        # Note: Spam messages usually aren't added to history
        return "<Response></Response>"

    # 3) Load Prefs Summary (Needed for Service Controller and Agent)
    prefs_summary = "No preferences recorded yet." # Default
    try:
        # This still relies on PreferenceLoaderTool reading the .txt file
        prefs_summary_loaded = preference_loader_tool.run(from_number)
        if prefs_summary_loaded and prefs_summary_loaded.strip():
            prefs_summary = prefs_summary_loaded.strip()
        else:
             logger.info(f"No preference summary loaded for {from_number} (file might be empty or only contain whitespace).")
    except Exception as e:
        logger.error(f"Error loading preference summary for {from_number}: {e}")

    # ---> 4) Customer Service Request Check <---
    matched_service_type = customer_service_matcher.match(incoming_msg, threshold=0.85)

    if matched_service_type:
        logger.info(f"Handling message as Customer Service Request: Type='{matched_service_type}'")
        service_result = customer_service_controller.handle_request(
            request_type=matched_service_type,
            message=incoming_msg,
            phone_number=from_number,
            prefs_summary=prefs_summary # Pass loaded preferences
        )
        response_text = service_result['response_message']

        # ---> IMPORTANT: Update history with this interaction <---
        updated_history = user_history_str + f"User: {incoming_msg}\nAI: {response_text.replace('^MSG^', ' ')}\n" # Store combined response
        conversation_histories[from_number] = updated_history
        logger.debug(f"History updated for {from_number} after service request handling.")

        # ---> Send response using MessageSenderTool <---
        try:
            logger.info(f"Sending customer service response to {from_number}...")
            message_sender_tool.run(to_number=from_number, message_body=response_text)
            logger.info(f"Customer service response sent to {from_number}.")
        except Exception as e:
            logger.exception(f"Error sending customer service message to {from_number}: {e}")

        # ---> Skip Agent and Profiler for direct service requests <---
        logger.info(f"Customer service request handled directly. Skipping agent and profiler for this turn.")
        return "<Response></Response>"

    # ---> 5) If NOT a service request, proceed with Agent <---
    logger.info(f"Message not classified as service request. Proceeding with Agent.")
    agent_memory = get_agent_memory(user_history_str)
    response_text = "Sorry, I had trouble processing that." # Default agent error response

    try:
        logger.info(f"Invoking agent for {from_number}...")
        agent_input = {
            "input": incoming_msg,
            "chat_history": agent_memory.chat_memory.messages,
            "phone_number": from_number,
            "preferences_summary": prefs_summary # Pass prefs to agent
        }
        logger.debug(f"Agent Input: {json.dumps(agent_input, default=str)}") # Log input safely
        response = agent_executor.invoke(agent_input)
        agent_output = response.get("output")
        if agent_output and agent_output.strip():
            response_text = agent_output.strip()
        else:
             logger.warning(f"Agent for {from_number} returned empty or None output. Response: {response}")
             response_text = "I'm sorry, I couldn't generate a response for that right now."

        logger.info(f"Agent execution completed for {from_number}.")

    except Exception as e:
        logger.exception(f"Error during agent execution for {from_number}: {e}")
        response_text = "I had trouble processing that request. Could you please try rephrasing?" # More specific error

    # ---> 6) Update History (Agent conversation) <---
    # Combine multi-bubble responses for storage if needed, or store raw output
    stored_ai_response = response_text.replace('^MSG^', ' ') # Simple combination for history log
    updated_history = user_history_str + f"User: {incoming_msg}\nAI: {stored_ai_response}\n"
    conversation_histories[from_number] = updated_history
    logger.debug(f"History updated for {from_number} after agent handling.")

    # ---> 7) Finalize Preferences using ProfilerAgent (After agent interaction) <---
    try:
        logger.info(f"Running preference profiling for {from_number}...")
        profiling_result = profiler_agent.profile_user(
            phone_number=from_number,
            conversation_history=updated_history # Use the *updated* history
        )
        logger.info(f"Preference profiling result for {from_number}: {profiling_result}")
    except Exception as e:
        logger.exception(f"Error during preference profiling for {from_number}: {e}")

    # ---> 8) Send Agent Response <---
    try:
        logger.info(f"Sending agent response to {from_number}...")
        message_sender_tool.run(to_number=from_number, message_body=response_text) # Send original agent output with ^MSG^
        logger.info(f"Agent response sent to {from_number}.")
    except Exception as e:
        logger.exception(f"Error sending agent message to {from_number}: {e}")

    return "<Response></Response>"

# --- Main Execution ---
if __name__ == "__main__":
    # Check if FAISS is available on startup
    if not faiss:
        logger.critical("FAISS library not found or failed to import. Application may not function correctly.")
    # Check if service matcher loaded correctly
    if customer_service_matcher.example_embeddings is None:
        logger.warning("Customer Service Matcher failed to initialize properly. Service request detection may not work.")

    logger.info("Starting Flask application...")
    # Use host="0.0.0.0" for Docker/network access, debug=False for production
    # Install scikit-learn: pip install scikit-learn
    app.run(host="0.0.0.0", port=55555, debug=True) # Keep debug=True for development