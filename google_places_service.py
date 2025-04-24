import logging
from typing import List, Dict, Any, Optional
import requests

from config import settings
from models.travel import PointOfInterest, TravelPreference

logger = logging.getLogger(__name__)

class GooglePlacesService:
    """
    Service for interacting with the Google Places API to fetch
    points of interest based on location and traveler preferences.
    """
    
    # Mapping of user preferences to Google Places types
    PREFERENCE_TO_PLACE_TYPE = {
        TravelPreference.CULTURE: ["museum", "art_gallery", "library", "tourist_attraction"],
        TravelPreference.RELAXATION: ["spa", "beauty_salon", "park"],
        TravelPreference.ADVENTURE: ["amusement_park", "tourist_attraction", "natural_feature"],
        TravelPreference.FOOD: ["restaurant", "cafe", "bakery", "bar"],
        TravelPreference.NATURE: ["park", "natural_feature", "campground"],
        TravelPreference.NIGHTLIFE: ["night_club", "bar", "casino"],
        TravelPreference.LUXURY: ["spa", "jewelry_store", "shopping_mall"],
        TravelPreference.BUDGET: ["restaurant", "tourist_attraction", "park"],
        TravelPreference.FAMILY: ["amusement_park", "aquarium", "zoo", "museum"],
        TravelPreference.SHOPPING: ["shopping_mall", "department_store", "clothing_store"],
        TravelPreference.BEACH: ["beach"],
        TravelPreference.MOUNTAIN: ["natural_feature", "campground"]
    }
    
    def __init__(self):
        self.api_key = settings.GOOGLE_PLACES_KEY
        self.base_url = "https://places.googleapis.com/v1"
        self.headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.priceLevel,places.rating,places.types,places.photos,places.editorialSummary"
        }
    
    def get_points_of_interest(self, city: str, preferences: List[str], 
                              max_per_preference: int = 3) -> List[PointOfInterest]:
        """
        Search for points of interest in a city based on traveler preferences.
        Returns a list of PointOfInterest objects.
        """
        results = []
        
        # If no preferences specified, use a default set
        if not preferences:
            preferences = [TravelPreference.CULTURE, TravelPreference.ADVENTURE, TravelPreference.FOOD]
        
        # Process each preference
        for preference in preferences:
            try:
                # Get place types for this preference
                place_types = []
                if hasattr(TravelPreference, preference):
                    pref_enum = getattr(TravelPreference, preference)
                    place_types = self.PREFERENCE_TO_PLACE_TYPE.get(pref_enum, [])
                else:
                    # Try to match string to enum
                    for enum_pref in TravelPreference:
                        if enum_pref.value.lower() == preference.lower():
                            place_types = self.PREFERENCE_TO_PLACE_TYPE.get(enum_pref, [])
                            break
                
                # If no matching preference found, use tourist_attraction as fallback
                if not place_types:
                    place_types = ["tourist_attraction"]
                
                # Search for each place type
                for place_type in place_types:
                    places = self._search_places_text(city, place_type, max_results=max_per_preference)
                    
                    if not places:
                        logger.warning(f"No places found for {place_type} in {city}")
                        continue
                        
                    for place in places:
                        # Create PointOfInterest object
                        display_name = place.get("displayName", {}).get("text", "Unknown")
                        poi = PointOfInterest(
                            name=display_name,
                            category=preference,
                            rating=place.get("rating", 4.0),
                            address=place.get("formattedAddress", f"{city}"),
                            description=self._get_place_description(place),
                            price_level=self._convert_price_level(place.get("priceLevel"))
                        )
                        
                        # Add photo URL if available
                        if "photos" in place and place["photos"]:
                            photo_name = place["photos"][0].get("name")
                            if photo_name:
                                poi.image_url = self._get_photo_url(photo_name)
                        
                        results.append(poi)
                        
                    # If we got enough places, move on to the next preference
                    if len(places) >= max_per_preference:
                        break
                        
            except Exception as e:
                logger.error(f"Error getting places for preference {preference}: {str(e)}")
                continue
        
        # If we got no results, return a few dummy POIs
        if not results:
            results = [
                PointOfInterest(
                    name=f"{city} Attraction {i}",
                    category="Tourist Attraction",
                    rating=4.0,
                    address=f"Main Street, {city}",
                    image_url="https://via.placeholder.com/400x300.png?text=No+Image+Available"
                ) for i in range(1, 4)
            ]
        
        return results
    
    def _convert_price_level(self, price_level_str: str) -> Optional[int]:
        """
        Convert Google Places API price level strings to integer values
        
        Args:
            price_level_str: String price level from API (e.g., "PRICE_LEVEL_MODERATE")
            
        Returns:
            Integer price level (0-4) or None if not recognized
        """
        if not price_level_str:
            return None
            
        price_level_mapping = {
            "PRICE_LEVEL_FREE": 0,
            "PRICE_LEVEL_INEXPENSIVE": 1,
            "PRICE_LEVEL_MODERATE": 2,
            "PRICE_LEVEL_EXPENSIVE": 3,
            "PRICE_LEVEL_VERY_EXPENSIVE": 4
        }
        
        return price_level_mapping.get(price_level_str)
    
    def _search_places_text(self, city: str, place_type: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search for places of a specific type in a city using the Places API Text Search.
        Returns processed place data from the Google Places API.
        """
        try:
            url = f"{self.base_url}/places:searchText"
            
            # Prepare the request body
            data = {
                "textQuery": f"{place_type} in {city}",
                "maxResultCount": max_results,
                "languageCode": "en"
            }
            
            response = requests.post(url, headers=self.headers, json=data, timeout=10)
            response.raise_for_status()
            response_data = response.json()
            
            if "places" not in response_data:
                logger.warning(f"No places found for {place_type} in {city}")
                return []
            
            return response_data.get("places", [])
            
        except requests.RequestException as e:
            logger.error(f"Error in Google Places API request: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in _search_places_text: {str(e)}")
            return []
    
    def _search_places_nearby(self, city: str, place_type: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search for places of a specific type near a location.
        This is an alternative method that can be used if we have location coordinates.
        """
        try:
            # First get the location of the city
            location_id = self._get_location_id(city)
            if not location_id:
                logger.warning(f"Could not get location ID for {city}")
                return []
                
            url = f"{self.base_url}/places:searchNearby"
            
            # Prepare the request body
            data = {
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "placeId": location_id
                        },
                        "radius": 5000.0  # 5km radius
                    }
                },
                "includedTypes": [place_type],
                "maxResultCount": max_results,
                "languageCode": "en"
            }
            
            response = requests.post(url, headers=self.headers, json=data, timeout=10)
            response.raise_for_status()
            response_data = response.json()
            
            if "places" not in response_data:
                logger.warning(f"No places found for {place_type} near {city}")
                return []
            
            return response_data.get("places", [])
            
        except requests.RequestException as e:
            logger.error(f"Error in Google Places nearby search API request: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in _search_places_nearby: {str(e)}")
            return []
            
    def _get_location_id(self, city: str) -> Optional[str]:
        """Get place ID for a city name using autocomplete."""
        try:
            url = f"{self.base_url}/places:autocomplete"
            
            data = {
                "textQuery": city,
                "languageCode": "en",
                "types": ["locality"]
            }
            
            response = requests.post(url, headers=self.headers, json=data, timeout=10)
            response.raise_for_status()
            response_data = response.json()
            
            if "places" in response_data and response_data["places"]:
                return response_data["places"][0].get("id")
            return None
            
        except Exception as e:
            logger.error(f"Error getting location ID for {city}: {str(e)}")
            return None
    
    def _get_place_description(self, place: Dict[str, Any]) -> Optional[str]:
        """Extract a description from place data if available."""
        # Try to get editorial summary first
        if "editorialSummary" in place and place["editorialSummary"]:
            return place["editorialSummary"].get("text")
        
        # Fallback to constructing a simple description from the place types
        if "types" in place and place["types"]:
            readable_types = [t.replace("_", " ").title() for t in place["types"][:3]]
            display_name = place.get("displayName", {}).get("text", "This place")
            return f"{display_name} is a {' and '.join(readable_types)}."
        
        return None
    
    def _get_photo_url(self, photo_name: str, max_width: int = 400) -> str:
        """Get a URL for a place photo using its name."""
        return f"{self.base_url}/{photo_name}/media?key={self.api_key}&maxWidthPx={max_width}"
    
    def _get_place_details(self, place_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a place using its place_id."""
        try:
            url = f"{self.base_url}/places/{place_id}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Error in Google Places details API request: {str(e)}")
            return None