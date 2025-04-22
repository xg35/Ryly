# my_project/google_map.py

import os
import logging
from typing import Type, List, Dict, Optional, Any
from pydantic import BaseModel, Field

# You'll need to install: pip install googlemaps
import googlemaps
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

# --- Input Schema for the Tool ---

class GooglePlacesInput(BaseModel):
    query: str = Field(description="The search query describing the place or atmosphere (e.g., 'romantic italian restaurant', 'lively bar near beach', 'mayan ruins tours')")
    location: str = Field(default="Cancun, Mexico", description="The central location for the search (defaults to Cancun)")
    max_results: int = Field(default=3, description="Maximum number of places to return reviews for")

# --- Google Places Tool Definition ---

class GooglePlacesTool(BaseTool):
    name: str = "search_google_places_reviews"
    description: str = (
        "Useful for finding information and recent reviews about places OUTSIDE the resort, like restaurants, bars, attractions, or shops in the Cancun area. "
        "Use this when a guest asks for recommendations for specific types of places, atmospheres (e.g., 'romantic', 'lively', 'quiet'), or activities available locally. "
        "Input should be a query describing the desired place/atmosphere and optionally a location."
    )
    args_schema: Type[BaseModel] = GooglePlacesInput
    gmaps: googlemaps.Client = None # Initialize later
    cancun_coords: Dict[str, float] = {'lat': 21.1619, 'lng': -86.8515} # Cache Cancun coords

    def __init__(self, api_key: str, **kwargs):
        # Initialize BaseTool first
        super().__init__(**kwargs)
        if not api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable must be set")
        try:
            self.gmaps = googlemaps.Client(key=api_key)
            # Optional: Geocode Cancun once if needed, but hardcoding is often fine for a fixed resort location
            # geocode_result = self.gmaps.geocode('Cancun, Mexico')
            # if geocode_result:
            #     self.cancun_coords = geocode_result[0]['geometry']['location']
            # else:
            #     logger.warning("Could not geocode Cancun, using default coordinates.")
            #     # Keep default coords if geocoding fails
        except Exception as e:
            logger.error(f"Failed to initialize Google Maps client: {e}")
            raise ConnectionError("Could not connect to Google Maps services.") from e

    def _run(self, query: str, location: str = "Cancun, Mexico", max_results: int = 3) -> str:
        """Finds places and retrieves recent reviews."""
        if not self.gmaps:
            return "Error: Google Maps client is not initialized."

        logger.info(f"Running Google Places search for query='{query}', location='{location}'")

        try:
            # 1. Find nearby places based on the query (as keyword)
            # If location is not Cancun, geocode it (optional, could stick to Cancun)
            search_location = self.cancun_coords
            if location.lower() != "cancun, mexico" and location.lower() != "cancun":
                 try:
                     geocode_result = self.gmaps.geocode(location)
                     if geocode_result:
                         search_location = geocode_result[0]['geometry']['location']
                         logger.info(f"Using geocoded location for '{location}': {search_location}")
                     else:
                         logger.warning(f"Could not geocode '{location}', defaulting to Cancun coordinates.")
                 except Exception as geo_e:
                     logger.warning(f"Geocoding error for '{location}': {geo_e}. Defaulting to Cancun.")


            places_result = self.gmaps.places_nearby(
                location=search_location,
                keyword=query,
                radius=10000,  # Search within 10km radius (adjust as needed)
                # rank_by='prominence' # Alternative: rank by prominence if radius not used
                # type='restaurant' # Optionally add type, but keyword is often better
            )

            results = places_result.get('results', [])
            if not results:
                logger.info(f"No Google Places results found for query: '{query}' near {location}")
                return f"Sorry, I couldn't find specific places matching '{query}' near {location} on Google Maps."

            output_parts = []
            processed_count = 0

            # 2. Get details (including reviews) for the top results
            for place in results:
                if processed_count >= max_results:
                    break

                place_id = place.get('place_id')
                if not place_id:
                    continue

                try:
                    details = self.gmaps.place(
                        place_id=place_id,
                        fields=['name', 'formatted_address', 'rating', 'reviews', 'url'] # Request specific fields
                    )
                    place_details = details.get('result')

                    if place_details:
                        name = place_details.get('name', 'N/A')
                        address = place_details.get('formatted_address', 'N/A')
                        rating = place_details.get('rating', 'N/A')
                        reviews = place_details.get('reviews', [])

                        place_info = f"Place: {name}\nRating: {rating}/5\nAddress: {address}"

                        review_snippets = []
                        if reviews:
                            # Sort reviews by time (newest first) and take top 1 or 2
                            reviews.sort(key=lambda r: r.get('time', 0), reverse=True)
                            for review in reviews[:2]: # Get max 2 newest reviews
                                author = review.get('author_name', 'A guest')
                                review_text = review.get('text', '').strip()
                                review_rating = review.get('rating', '')
                                if review_text: # Only add if text exists
                                    review_snippets.append(f"- Review ({review_rating}/5) by {author}: \"{review_text[:150]}...\"") # Truncate long reviews

                        if review_snippets:
                            place_info += "\nRecent Reviews:\n" + "\n".join(review_snippets)
                        else:
                            place_info += "\nRecent Reviews: (Not available)"

                        output_parts.append(place_info)
                        processed_count += 1

                except googlemaps.exceptions.ApiError as detail_e:
                    logger.error(f"Google Places API error getting details for place_id {place_id}: {detail_e}")
                except Exception as detail_e:
                    logger.error(f"Error processing place details for place_id {place_id}: {detail_e}")

            if not output_parts:
                 return f"Found places matching '{query}' but couldn't retrieve details or reviews."

            return "\n\n---\n\n".join(output_parts) # Separate places clearly

        except googlemaps.exceptions.ApiError as e:
            logger.error(f"Google Places API error during search: {e}")
            return f"There was an issue searching Google Places: {e}"
        except Exception as e:
            logger.exception(f"Unexpected error during Google Places search: {e}") # Log full traceback
            return "An unexpected error occurred while searching Google Places."

    async def _arun(self, query: str, location: str = "Cancun, Mexico", max_results: int = 3) -> str:
        # Google Maps client library is primarily synchronous.
        # For a truly async version, you'd need an async HTTP client (like httpx or aiohttp)
        # and manually construct the API requests, or use loop.run_in_executor.
        # For simplicity, we run the sync version in the default executor.
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, query, location, max_results)