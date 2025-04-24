import time
import logging
from datetime import date, datetime
from typing import List, Optional, Dict, Any, Tuple

import requests
from fastapi import HTTPException

from config import settings
from models.travel import FlightOption, HotelOption

logger = logging.getLogger(__name__)

class AmadeusService:
    """
    Service for interacting with the Amadeus API to fetch flight and hotel information.
    Uses Amadeus test mode by default, which returns sandbox data.
    """
    
    # City code mapping for common destinations
    IATA_OVERRIDES = {
        "singapore": "SIN",
        "tokyo": "TYO",
        "paris": "PAR",
        "london": "LON",
        "new york": "NYC",
        "bangkok": "BKK",
        "dubai": "DXB",
        "sydney": "SYD",
        "san francisco": "SFO",
        "los angeles": "LAX",
    }
    
    def __init__(self):
        self.key = settings.AMADEUS_API_KEY
        self.secret = settings.AMADEUS_API_SECRET
        self.test_mode = settings.AMADEUS_TEST_MODE
        
        # Set the appropriate base URL based on mode
        if self.test_mode:
            self.base_url = "https://test.api.amadeus.com"
        else:
            self.base_url = "https://api.amadeus.com"
            
        # Authentication state
        self._token = None
        self._expiry = 0.0
        
        # Create a session for connection pooling
        self.session = requests.Session()
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get or refresh the authentication token and return headers."""
        if not self._token or time.time() >= self._expiry:
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/security/oauth2/token",
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.key,
                        "client_secret": self.secret
                    }
                )
                response.raise_for_status()
                data = response.json()
                self._token = data["access_token"]
                self._expiry = time.time() + data["expires_in"] - 60  # Refresh 1 minute before expiry
                
            except requests.RequestException as e:
                logger.error(f"Amadeus authentication error: {str(e)}")
                raise HTTPException(status_code=503, detail="Travel service temporarily unavailable")
        
        return {"Authorization": f"Bearer {self._token}"}
    
    def get_city_code(self, city: str) -> Optional[str]:
        """Convert a city name to an IATA city code."""
        city_lower = city.strip().lower()
        
        # Check override map first for common cities
        if city_lower in self.IATA_OVERRIDES:
            return self.IATA_OVERRIDES[city_lower]
        
        try:
            # Otherwise search the Amadeus location API
            response = self.session.get(
                f"{self.base_url}/v1/reference-data/locations",
                params={"keyword": city, "subType": "CITY"},
                headers=self._get_auth_headers()
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            
            if data:
                return data[0].get("iataCode")
            return None
            
        except requests.RequestException as e:
            logger.error(f"Error fetching city code for {city}: {str(e)}")
            return None
    
    def get_flight_offers(self, origin: str, destination: str, 
                         depart_date: date, return_date: date,
                         adults: int = 1, max_results: int = 5) -> List[FlightOption]:
        """
        Search for flight offers between cities on specified dates.
        Returns a list of FlightOption objects.
        """
        # Get city codes
        origin_code = self.get_city_code(origin)
        dest_code = self.get_city_code(destination)
        
        if not origin_code or not dest_code:
            raise ValueError(f"Could not find city codes for {origin} or {destination}")
        
        try:
            # Query the flight offers search API
            response = self.session.get(
                f"{self.base_url}/v2/shopping/flight-offers",
                params={
                    "originLocationCode": origin_code,
                    "destinationLocationCode": dest_code,
                    "departureDate": depart_date.isoformat(),
                    "returnDate": return_date.isoformat(),
                    "adults": adults,
                    "max": max_results,
                    "currencyCode": "SGD"
                },
                headers=self._get_auth_headers()
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            
            # Convert to FlightOption objects
            flight_options = []
            for flight in data:
                try:
                    airline_code = flight.get("validatingAirlineCodes", ["Unknown"])[0]
                    price = float(flight["price"]["grandTotal"])
                    
                    # Get some basic flight details from the first segment
                    segments = flight.get("itineraries", [{}])[0].get("segments", [{}])
                    flight_number = segments[0].get("number", "") if segments else ""
                    
                    departure_time = segments[0].get("departure", {}).get("at", "") if segments else ""
                    arrival_time = segments[-1].get("arrival", {}).get("at", "") if segments else ""
                    
                    flight_options.append(FlightOption(
                        airline=airline_code,
                        price=price,
                        origin=origin,
                        destination=destination,
                        depart_date=depart_date,
                        return_date=return_date,
                        flight_number=flight_number,
                        departure_time=departure_time,
                        arrival_time=arrival_time
                    ))
                except (KeyError, IndexError) as e:
                    logger.warning(f"Error parsing flight data: {str(e)}")
                    continue
            
            # If we're in test mode and got no results, return a dummy flight
            if not flight_options and self.test_mode:
                flight_options = [
                    FlightOption(
                        airline="SQ",
                        price=800.0,
                        origin=origin,
                        destination=destination,
                        depart_date=depart_date,
                        return_date=return_date,
                        flight_number="SQ123",
                        departure_time="09:00",
                        arrival_time="14:00"
                    )
                ]
            #print(flight_options)
            return flight_options
            
        except requests.RequestException as e:
            logger.error(f"Error fetching flight offers: {str(e)}")
            if self.test_mode:
                # Return dummy data in test mode
                return [
                    FlightOption(
                        airline="SQ",
                        price=800.0,
                        origin=origin,
                        destination=destination,
                        depart_date=depart_date,
                        return_date=return_date
                    )
                ]
            raise HTTPException(status_code=503, detail="Flight search service temporarily unavailable")
    
    def search_hotels_by_city(self, city: str, radius: int = 20) -> List[Dict[str, Any]]:
        """
        Search for hotels in a city using the reference-data/locations/hotels/by-city API.
        Returns raw hotel data from the API.
        """
        city_code = self.get_city_code(city)
        if not city_code:
            raise ValueError(f"Could not find city code for {city}")
        
        try:
            response = self.session.get(
                f"{self.base_url}/v1/reference-data/locations/hotels/by-city",
                params={
                    "cityCode": city_code, 
                    "radius": radius, 
                    "radiusUnit": "KM",
                    "hotelSource": "ALL"
                },
                headers=self._get_auth_headers()
            )
            response.raise_for_status()
            result = response.json()
            
            # Log the response for debugging
            logger.debug(f"Hotel search response: {result}")
            #print the top 10 values of the result
            #print(result.get("data", [])[:10])
            return result.get("data", [])
            
        except requests.RequestException as e:
            logger.error(f"Error searching hotels by city {city}: {str(e)}")
            if self.test_mode:
                # Return dummy hotel data in test mode
                return [
                    {
                        "hotelId": f"DUMMY_HOTEL_{i}",
                        "name": f"Demo Hotel {i}",
                        "chainCode": "DH",
                        "geoCode": {"latitude": 0, "longitude": 0},
                        "address": {"countryCode": city_code}
                    } for i in range(1, 6)
                ]
            raise HTTPException(status_code=503, detail="Hotel search service temporarily unavailable")
    
    def search_hotels_by_geocode(self, latitude: float, longitude: float, radius: int = 20) -> List[Dict[str, Any]]:
        """
        Search for hotels near a specific geocode using the reference-data API.
        Returns raw hotel data from the API.
        """
        try:
            response = self.session.get(
                f"{self.base_url}/v1/reference-data/locations/hotels/by-geocode",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "radius": radius,
                    "radiusUnit": "KM"
                },
                headers=self._get_auth_headers()
            )
            response.raise_for_status()
            result = response.json()
            
            # Log the response for debugging
            logger.debug(f"Geocode hotel search response: {result}")
            
            return result.get("data", [])
            
        except requests.RequestException as e:
            logger.error(f"Error searching hotels by geocode: {str(e)}")
            if self.test_mode:
                # Return dummy hotel data in test mode
                return [
                    {
                        "hotelId": f"DUMMY_HOTEL_{i}",
                        "name": f"Demo Hotel {i}",
                        "chainCode": "DH",
                        "geoCode": {"latitude": latitude, "longitude": longitude},
                        "address": {"countryCode": "FR"}
                    } for i in range(1, 6)
                ]
            raise HTTPException(status_code=503, detail="Hotel search service temporarily unavailable")
    
    def get_hotel_offers(self, city: str, checkin_date: date, checkout_date: date,
                       adults: int = 1, rooms: int = 1, max_results: int = 10) -> List[HotelOption]:
        """
        Search for hotel offers in a city on specified dates.
        First retrieves hotels in the city using reference-data API, then
        converts them to HotelOption objects with estimated pricing.
        
        Returns a list of HotelOption objects.
        """
        try:
            # Step 1: Get hotels located in the city
            hotels_in_city = self.search_hotels_by_city(city)
            
            # If no hotels found, return dummy data in test mode or empty list
            if not hotels_in_city:
                if self.test_mode:
                    return [
                        HotelOption(
                            name=f"Demo Hotel {i}",
                            price_per_night=100 + i * 50,
                            stars=min(5, 3 + i * 0.5),
                            city=city,
                            address=f"{i+1} Main Street, {city}",
                            hotel_id=f"DUMMY_{i}"
                        ) for i in range(3)
                    ]
                else:
                    return []
            
            logger.info(f"Found {len(hotels_in_city)} hotels in {city}")
            
            # Step 2: Convert to HotelOption objects (limited to max_results)
            hotel_options = []
            
            for hotel in hotels_in_city[:max_results]:
                try:
                    # Extract data from the hotel information
                    hotel_id = hotel.get("hotelId", "")
                    name = hotel.get("name", "Unknown Hotel")
                    
                    # Some hotels may not have chain codes, so default to "IND" (Independent)
                    chain_code = hotel.get("chainCode", "IND")
                    
                    # Get address information if available
                    address_info = hotel.get("address", {})
                    country_code = address_info.get("countryCode", "")
                    
                    # Format address (simple version)
                    address = f"{country_code}"
                    
                    # Get distance information if available
                    distance_info = hotel.get("distance", {})
                    distance = distance_info.get("value", 0)
                    
                    # Get ratings - in a real implementation, you might get this from another source
                    # For now, generate a rating between 2 and 5
                    import hashlib
                    # Use hotel_id to generate a consistent but "random" rating
                    hash_val = int(hashlib.md5(hotel_id.encode()).hexdigest(), 16) % 30
                    stars = 2.0 + (hash_val / 10.0)  # Rating between 2.0 and 5.0
                    
                    # Generate a price based on stars
                    base_price = 80 + (stars * 40)  # Higher star rating = higher price
                    
                    # Add a slight random variation
                    import random
                    price_variance = random.uniform(0.8, 1.2)
                    price_per_night = base_price * price_variance
                    
                    hotel_options.append(HotelOption(
                        name=name,
                        price_per_night=price_per_night,
                        stars=stars,
                        city=city,
                        address=address,
                        hotel_id=hotel_id,
                        chain_code=chain_code,
                        distance=distance
                    ))
                except Exception as e:
                    logger.warning(f"Error processing hotel data: {str(e)}")
                    continue
            
            # If we couldn't convert any hotels, but we had results, something went wrong
            # In test mode, return dummy data; otherwise return an empty list
            if not hotel_options and hotels_in_city and self.test_mode:
                hotel_options = [
                    HotelOption(
                        name=f"Demo Hotel {i}",
                        price_per_night=100 + i * 50,
                        stars=min(5, 3 + i * 0.5),
                        city=city,
                        address=f"{i+1} Main Street, {city}"
                    ) for i in range(3)
                ]
            
            return hotel_options
            
        except Exception as e:
            logger.error(f"Error fetching hotel offers: {str(e)}")
            if self.test_mode:
                # Return dummy data in test mode
                return [
                    HotelOption(
                        name=f"Demo Hotel {i}",
                        price_per_night=100 + i * 50,
                        stars=min(5, 3 + i * 0.5),
                        city=city
                    ) for i in range(3)
                ]
            raise HTTPException(status_code=503, detail="Hotel search service temporarily unavailable")

    def get_hotel_details(self, hotel_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific hotel.
        In a real implementation, this would call the hotel details API.
        For now, we'll stub it out with basic info.
        """
        try:
            # In a real implementation, you would call an API like:
            # response = self.session.get(
            #     f"{self.base_url}/v1/reference-data/locations/hotels/{hotel_id}",
            #     headers=self._get_auth_headers()
            # )
            
            # For now, just return some made-up details based on the hotel ID
            if self.test_mode or not hotel_id.startswith("DUMMY"):
                import hashlib
                # Use hotel_id to generate a consistent but "random" rating
                hash_val = int(hashlib.md5(hotel_id.encode()).hexdigest(), 16)
                
                # Generate some amenities
                amenities = []
                possible_amenities = [
                    "Free Wi-Fi", "Swimming Pool", "Fitness Center", "Restaurant",
                    "Bar/Lounge", "Room Service", "Business Center", "Spa", 
                    "Airport Shuttle", "Breakfast Included", "Parking", "Concierge",
                    "Pet Friendly", "Laundry Service", "Meeting Rooms"
                ]
                
                # Select some amenities based on the hotel_id hash
                for i, amenity in enumerate(possible_amenities):
                    if (hash_val >> i) & 1:
                        amenities.append(amenity)
                
                # If we ended up with no amenities, add Wi-Fi at minimum
                if not amenities:
                    amenities = ["Free Wi-Fi"]
                
                # Generate a description
                descriptions = [
                    "A luxurious retreat in the heart of the city.",
                    "Experience comfort and convenience at this modern hotel.",
                    "A charming property offering a unique local experience.",
                    "Elegant accommodations with world-class service.",
                    "A traveler's haven with all the essential amenities."
                ]
                description = descriptions[hash_val % len(descriptions)]
                
                return {
                    "hotel_id": hotel_id,
                    "description": description,
                    "amenities": amenities,
                    "check_in_time": "14:00",
                    "check_out_time": "12:00",
                    "photos": [
                        "https://example.com/hotel-photos/lobby.jpg",
                        "https://example.com/hotel-photos/room.jpg",
                        "https://example.com/hotel-photos/restaurant.jpg"
                    ],
                    "policies": {
                        "cancellation": "Free cancellation up to 24 hours before check-in.",
                        "children": "Children of all ages are welcome.",
                        "pets": "Pets are not allowed."
                    }
                }
        except Exception as e:
            logger.error(f"Error fetching hotel details: {str(e)}")
            return {
                "hotel_id": hotel_id,
                "description": "Information temporarily unavailable",
                "amenities": ["Free Wi-Fi"],
                "check_in_time": "14:00",
                "check_out_time": "12:00"
            }