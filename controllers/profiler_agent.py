# controllers/profiler_agent.py

import json
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class ProfilerAgent:
    def __init__(self, llm, get_prefs_filename, file_storage_helper, get_user_preferences_store):
        self.llm = llm
        self.get_prefs_filename = get_prefs_filename
        self.file_storage_helper = file_storage_helper
        self.get_user_preferences_store = get_user_preferences_store

        # Define the prompt template once during initialization
        template_text = """
        You are an expert **Client Insights Analyst** for a luxury resort, focusing on deep understanding of guest needs like a personal concierge.
        Your task is to analyze the *entire* time-stamped conversation history provided below ([timestamp] Speaker: text).
        Synthesize this history into a **single, actionable JSON profile**, capturing preferences, requirements, context, and **when** requests/notes were made.

        Follow these [Instructions] meticulously, interpreting the user's statements within the [Categories] provided.

        [Categories] - Interpret these broadly to capture relevant nuances:
        [category-1] customer_name - The guest's name. Identify the most likely name if mentioned. (No timestamp needed)
        [category-2] family_members - Names or descriptions of companions. Note relationships. (No timestamp needed)
        [category-3] arrival_date - Confirmed or strongly indicated arrival date. (No timestamp needed)
        [category-4] departure_date - Confirmed or strongly indicated departure date. (No timestamp needed)
        [category-5] likes - **Significant Preferences & Affinities:** Specific items/experiences the guest expressed strong positive preference for. Include *why* if stated. (No timestamp needed for general likes).
        [category-6] dislikes - **Significant Aversions & Dislikes:** Specific items/services/experiences the guest explicitly dislikes or wants to avoid. Include allergies/restrictions if framed as dislikes. (No timestamp needed for general dislikes).
        [category-7] requests - **Actionable Needs & Intentions w/ Timestamp:** Specific actions/items the guest asked for (e.g., 'extra towels', 'book dinner'). Capture the nature of the request *and* the timestamp from the corresponding `User:` line.
        [category-8] notes - **Contextual & Supporting Details w/ Timestamp:** Other crucial details like special occasions, purpose of visit, general interests, allergies (not framed as dislikes), etc. Capture the detail *and* the timestamp from the corresponding `User:` line.
        [/Categories]

        [Instructions]
        1.  **Deep Reading:** Read the *entire* [transcripts] carefully, paying attention to the `[timestamp]` prefix on each line, nuance, emphasis, and changes over time.
        2.  **User Focus:** Extract information *only* from the 'User:' lines.
        3.  **Synthesize Holistically & Persistently:** Combine information from the *whole conversation* to build a cumulative profile. Retain all previously established preferences (likes, dislikes, etc.) unless explicitly contradicted, modified, or retracted by the user later. If a contradiction/update occurs, prioritize the *most recent* statement. For `requests` and `notes`, treat each instance as potentially unique based on its timestamp.
        4.  **Timestamp Extraction:** For every item added to the `requests` or `notes` arrays, extract the exact timestamp (e.g., `YYYY-MM-DD HH:MM:SS UTC`) found within the square brackets `[]` at the beginning of the 'User:' line where the request or note was stated.
        5.  **Handle Duplicates (Requests/Notes):** If the user states the *exact same* request or note text at *different times*, create a **separate JSON object** in the corresponding array for *each instance*, each with its own correct timestamp. Do *not* merge them or just update the timestamp. This explicitly tracks repeated requests/mentions.
        6.  **Discern Significance:** Focus on extracting information that is **actionable** or reveals a core aspect of the guest's desired experience. Filter out conversational filler.
        7.  **Generate Complete JSON:** Create a *single, complete* JSON object reflecting the *entire current understanding*.
        8.  **Strict JSON Structure:** Use the following JSON structure. **Pay close attention to the structure of `requests` and `notes` arrays - they contain objects.**
            ```json
            {{  // <-- Escaped Outer Brace
              "customer_name": "string",
              "family_members": ["string"],
              "arrival_date": "string",
              "departure_date": "string",
              "likes": ["string"],
              "dislikes": ["string"],
              "requests": [
                {{ // <-- Escaped Inner Brace
                  "text": "string",
                  "timestamp": "YYYY-MM-DD HH:MM:SS UTC"
                }} // <-- Escaped Inner Brace
              ],
              "notes": [
                {{ // <-- Escaped Inner Brace
                  "text": "string",
                  "timestamp": "YYYY-MM-DD HH:MM:SS UTC"
                }} // <-- Escaped Inner Brace
              ]
            }}  // <-- Escaped Outer Brace
            ```
        9.  **Uniqueness & Consolidation (Non-Timestamped Fields):** Ensure `family_members`, `likes`, `dislikes` arrays contain unique string values. Consolidate similar items within *these* fields where appropriate. **Do NOT consolidate `requests` or `notes` objects - preserve each instance.**
        10. **Handle Missing Info:** If no relevant information is found for a key, use `""` for top-level strings or `[]` for arrays. If a request/note object cannot be formed (e.g., missing timestamp in history), omit that specific entry.
        11. **Raw JSON Output:** Output *only* the raw, valid JSON object. No explanations, apologies, or markdown.
        [/Instructions]

        [transcripts]:
        {conversation} // <-- This is the ONLY intended input variable

        Output only valid JSON:
        """
        # Ensure only 'conversation' is seen as input variable
        self.prompt = PromptTemplate(input_variables=["conversation"], template=template_text.strip())
        # Ensure the LLMChain uses the correctly parsed prompt
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)


    def profile_user(self, phone_number, conversation_history):
        """
        Analyzes the full conversation history, generates a complete preference profile,
        and overwrites the user's preference file if the profile has changed.
        """
        logger.info(f"Starting full preference profiling for {phone_number} based on history.")
        logger.debug(f"History for profiler:\n{conversation_history}") # Log the history being sent

        generated_profile_str = None
        try:
            logger.debug(f"Invoking LLM chain for preference extraction for {phone_number}.")
            # Prepare the input dictionary correctly
            input_dict = {"conversation": conversation_history}
            logger.debug(f"Input keys provided to chain: {list(input_dict.keys())}")

            # Use invoke
            llm_response = self.chain.invoke(input_dict)
            generated_profile_str = llm_response.get('text', None) if isinstance(llm_response, dict) else str(llm_response)

            if not generated_profile_str:
                logger.error(f"LLM chain for {phone_number} returned empty or unexpected response: {llm_response}")
                return "Error during preference extraction: Empty LLM response."

            logger.debug(f"LLM raw output for {phone_number}: {generated_profile_str}")

        except ValueError as ve:
             # Catch the specific ValueError related to input keys
             logger.error(f"ValueError invoking LLM chain for {phone_number}: {ve}. Check prompt template and input keys.", exc_info=True)
             return f"Error during preference extraction: Prompt/Input mismatch ({ve})"
        except Exception as e:
            logger.error(f"Unexpected error invoking LLM chain for {phone_number}: {e}", exc_info=True)
            return "Error during preference extraction (LLM Call)."

        # --- JSON Parsing and File Writing ---
        try:
            if generated_profile_str.startswith("```json"): generated_profile_str = generated_profile_str[7:]
            if generated_profile_str.endswith("```"): generated_profile_str = generated_profile_str[:-3]
            generated_profile_str = generated_profile_str.strip()

            if not generated_profile_str.startswith("{") or not generated_profile_str.endswith("}"):
                logger.warning(f"LLM output for {phone_number} doesn't look like JSON: {generated_profile_str}")
                raise json.JSONDecodeError("Output doesn't start/end with braces.", generated_profile_str, 0)

            new_complete_profile = json.loads(generated_profile_str)
            logger.info(f"Successfully parsed generated profile for {phone_number}.")

        except json.JSONDecodeError as e:
            logger.error(f"LLM did not return valid JSON for {phone_number}. Error: {e}. Output was: {generated_profile_str}", exc_info=True)
            return "Profiler failed to generate valid preference data."
        except Exception as e:
             logger.error(f"Unexpected error processing LLM output for {phone_number}: {e}", exc_info=True)
             return "Unexpected error processing profile data."

        # --- File Handling ---
        filename = self.get_prefs_filename(phone_number) # Gets .txt currently
        logger.debug(f"Attempting to write profile to: {filename}")
        existing_profile = self.file_storage_helper.load_json(filename) # Reads JSON (returns {} if file is .txt or non-existent)

        if json.dumps(new_complete_profile, sort_keys=True) != json.dumps(existing_profile, sort_keys=True):
            logger.info(f"Preference profile changed for {phone_number}. Overwriting file: {filename}")
            try:
                # This writes JSON data to the file specified by filename (currently .txt)
                self.file_storage_helper.write_json(filename, new_complete_profile)
                logger.info(f"Successfully wrote updated preferences to {filename}.")
                return new_complete_profile # Return the data that was written
            except Exception as e:
                 logger.error(f"Failed to write updated preferences to {filename}: {e}", exc_info=True)
                 return "Error saving updated preferences."
        else:
            logger.info(f"No changes detected in preference profile for {phone_number}.")
            # Even if no changes, the file might not exist yet for a new user.
            # Let's ensure the file is created on the first successful run.
            if not os.path.exists(filename):
                 logger.info(f"File {filename} does not exist. Writing initial profile.")
                 try:
                      self.file_storage_helper.write_json(filename, new_complete_profile)
                      logger.info(f"Successfully wrote initial preferences to {filename}.")
                      return new_complete_profile
                 except Exception as e:
                      logger.error(f"Failed to write initial preferences to {filename}: {e}", exc_info=True)
                      return "Error saving initial preferences."
            else:
                 return "No preference changes identified." # Return only if file already exists and no changes