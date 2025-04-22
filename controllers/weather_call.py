import requests
from typing import Dict, Optional

class WeatherController:
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(self):
        self.geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    
    def get_coordinates(self, city: str) -> Optional[Dict[str, float]]:
        """Get latitude/longitude for a city name"""
        try:
            response = requests.get(
                self.geocode_url,
                params={"name": city, "count": 1, "format": "json"}
            )
            response.raise_for_status()
            data = response.json()
            if data.get("results"):
                return {
                    "latitude": data["results"][0]["latitude"],
                    "longitude": data["results"][0]["longitude"]
                }
            return None
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None

    def get_weather(self, latitude: float, longitude: float, 
                  hourly: list = None, daily: list = None,
                  forecast_days: int = 1) -> Optional[Dict]:
        """Get current weather and forecast data"""
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,wind_speed_10m",
                "timezone": "auto"
            }
            
            if hourly:
                params["hourly"] = ",".join(hourly)
            if daily:
                params["daily"] = ",".join(daily)
                params["forecast_days"] = forecast_days
                
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Weather API error: {e}")
            return None
