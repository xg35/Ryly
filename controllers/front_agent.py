# my_project/controllers/front_agent.py

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from controllers.weather_call import WeatherController
from controllers.tavily import research # Added Tavily import
import datetime
import logging # Added for logging Tavily errors

# Configure logging for this module if not already configured elsewhere
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class FrontAgent:
    """
    Main agent for generating a response:
      - Reads from conversation history
      - Retrieves context from the facilities, user preferences, best practices
      - Combines them with the user's known preferences
      - Returns a final answer
    """
    def __init__(self, facilities_vector_store, best_practices_vector_store, llm,
                 get_user_preferences_store, get_prefs_filename, file_storage_helper):
        self.facilities_vector_store = facilities_vector_store
        self.best_practices_vector_store = best_practices_vector_store
        self.llm = llm
        self.get_user_preferences_store = get_user_preferences_store
        self.get_prefs_filename = get_prefs_filename
        self.file_storage_helper = file_storage_helper

        self.memory = ConversationBufferMemory(return_messages=True)
        self.prompt = PromptTemplate(
            input_variables=["history", "user_query", "context", "preferences"],
            template="""
            You are Ryly, a digital butler at a luxury resort in Cancun. Follow the [instructions] and [steps] with [Conversations], [Conversation-Flow] [Customer Preferences], [Relevant Info] and [User says] provided.

            [Definitions]
            [Instructions] Guidelines as to how you should response
            [Steps] Steps that you will follow in your analysis and execution
            [Conversations] Past conversation with the guest
            [Customer Preferences] Preferences that the guest has been previously established
            [Relevant Info] Information about the resort hotel facility
            [User says] The guest current statement or question
            [Conversation-Flow] Responding framework
            [Guidelines] Instructions on how to handle certain topics
            [/Definitions]

            [Steps]
            [Step-1]Review [Conversations] and [User says], establish what the guest needs or questions are[/Step-1]
            [Step-2]Review [Relevant Info] for answers first to assist Guest. [Web Search Results (from Tavily)] contains helpful information from external sources."[/Step-2]
            [Step-3]Assess [Customer Preferences] and select the optimal [Conversation-Flow] method to apply.[/Step-3]
            [Step-4]If answer is direct without further clarification or information, response immediately. If not, follow Step-5 [/Step-4]
            [Step-5]If there are more than 1 clear answer, derive question that you can ask the guest first (help them narrow their options)[/Step-5]
            [Step-6]Consider [Instructions] and craft your response[/Step-6]
            [Step-7]If you cannot find an answer; exit and response with "I can't help with this right now[/step-7]
            [/Steps]

            [Guidelines]
            [Guideline-1]When asked for instruction on how to go somewhere, consider asking the guest where they are now. Be clear on instruction such as, turn left, turn right or go straight or guide them from location to location. [/Guideline-1]
            [Guideline-2]When asked about recommendation on activities and exurisions, consider the weather to make suitable recommendation [/Guideline-2]
            [Guideline-3]When guest asks whats there to do or sound bored, consider the time and weather and propose activities (indoor or by the beach) [/Guideline-3]
            # [Guideline-1] [/Guideline-1]
            [/Guidelines]

            [Conversation-Flow]
            [Must Assess User Intent] Quickly interpret the user’s request.
            If the request is unclear or ambiguous, proceed to [Ask Clarifying Question(s)].
            Otherwise, decide between recommendation [Offer Recommendation then Ask] or [Answer Directly].
            [/Must Assess User Intent]

            [Ask Clarifying Question(s)]
            If you can’t fulfill the request because information is missing, ask specific clarifying question(s).
            Use short, dependency-focused sentence.
            [/Ask Clarifying Question(s)]

            [Offer Recommendation then Ask]
            If the user’s request suggests multiple or personalized options, propose the best fit first.
            Then, confirm alignment with a quick question (e.g., “Does that match what you’re looking for?”).
            [/Offer Recommendation then Ask]
            
            [Answer Directly]
            If the request is straightforward and you have a single clear answer, respond immediately.
            Use short dependency-linked phrases to keep the flow precise.
            [/Answer Directly]
            /Conversation-Flow]
            
            [Instructions]
            [instruction-1]Maintain a conversational and joyful yet professional tone.[/instruction-1] 
            [instruction-2]Use the dependency grammar linguistic framework rather than phrase structure grammar for the output. The idea is that the closer together each pair of words you’re connecting are, the easier the copy will be to comprehend.[/instruction-2] 
            [instruction-3]Break messages naturally, around 2–4 sentences per bubble.[/instruction-3] 
            [instruction-4]Use the delimiter ^MSG^ between bubbles.[/instruction-4] 
            [instruction-5]Provide helpful and concise responses without unnecessary commentaries.[/instruction-5] 
            [/Instructions]

            [Conversations]
            {history}
            [/Conversations]

            [Customer Preferences]
            {preferences}
            [/Customer Preferences]

            [Relevant Info] (from Facilities, Preferences, Best Practices):
            {context}
            [/Relevant Info]

            [User says] 
            {user_query}
            [/User says]
            
            Ryly's multi-message response (using ^MSG^ to separate):
            """.strip()
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, verbose=False)
        self.weather_controller = WeatherController()

    def load_customer_preferences_text(self, phone_number):
        filename = self.get_prefs_filename(phone_number)
        return "\n".join(self.file_storage_helper.load_lines(filename))

    def retrieve_context(self, user_query, phone_number):
        context_parts = []
        
        # Add current Cancun time (UTC-5)
        current_time_utc = datetime.datetime.utcnow()
        cancun_time = current_time_utc - datetime.timedelta(hours=5)
        context_parts.append(f"Current Cancun Date/Time: {cancun_time.strftime('%Y-%m-%d %H:%M:%S')} (UTC-5)")

        # 1) Facilities-based retrieval
        fac_docs = self.facilities_vector_store.similarity_search(user_query, k=5)
        if fac_docs:
            facilities_info = "\n".join(doc.page_content for doc in fac_docs)
            context_parts.append(f"Facilities Info:\n{facilities_info}")

        # 2) Preference-based retrieval (unique to the user)
        user_prefs_store = self.get_user_preferences_store(phone_number)
        pref_docs = user_prefs_store.similarity_search(user_query, k=5)
        if pref_docs:
            preferences_info = "\n".join(doc.page_content for doc in pref_docs)
            context_parts.append(f"Preferences Info:\n{preferences_info}")

        # 3) Best practices-based retrieval
        bp_docs = self.best_practices_vector_store.similarity_search(user_query, k=2)
        if bp_docs:
            bp_info = "\n".join(doc.page_content for doc in bp_docs)
            context_parts.append(f"Best Practices Info:\n{bp_info}")
            
        # 4) Add weather context for Cancun
        coords = self.weather_controller.get_coordinates("Cancun")
        if coords:
            weather_data = self.weather_controller.get_weather(coords["latitude"], coords["longitude"], 
                hourly=["temperature_2m", "relative_humidity_2m"],
                daily=["temperature_2m_max", "temperature_2m_min", "precipitation_probability_max"],
                forecast_days=8)
            
            if weather_data:
                context_parts.append(
                    f"Current Cancun Weather:\n"
                    f"- Temperature: {weather_data['current']['temperature_2m']}°C\n"
                    f"- Wind Speed: {weather_data['current']['wind_speed_10m']} km/h\n"
                    f"Today's Hourly Forecast:\n" + "\n".join(
                        f"{time.split('T')[1][:5]}: {temp}°C, {humidity}% humidity"
                        for time, temp, humidity in zip(
                            weather_data['hourly']['time'][:24],  # First 24 hours
                            weather_data['hourly']['temperature_2m'][:24],
                            weather_data['hourly']['relative_humidity_2m'][:24]
                        )
                    ) + "\n\n7-Day Forecast:\n" + "\n".join(
                        f"{date}: {tmax}°C/{tmin}°C, {rain}% rain"
                        for date, tmax, tmin, rain in zip(
                            weather_data['daily']['time'][1:8],  # Next 7 days after today
                            weather_data['daily']['temperature_2m_max'][1:8],
                            weather_data['daily']['temperature_2m_min'][1:8],
                            weather_data['daily']['precipitation_probability_max'][1:8]
                        )
                    )
                )

        # 5) Add Tavily web search results
        try:
            tavily_result = research(user_query, include_sources=True)
            #print(f"DEBUG: Tavily Result for query '{user_query}': {tavily_result}") # Added debug print
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
            # else: No answer found, don't add anything
        except Exception as e:
            logger.error(f"Error calling Tavily research for query '{user_query}': {str(e)}")


        if not context_parts:
            return "No relevant context found."
        print(context_parts)
        return "\n\n".join(context_parts)

    def generate_response(self, user_query, history_str, phone_number):
        customer_prefs = self.load_customer_preferences_text(phone_number)
        context = self.retrieve_context(user_query, phone_number)

        response = self.chain.run(
            history=history_str,
            user_query=user_query,
            context=context,
            preferences=customer_prefs
        )
        return response
