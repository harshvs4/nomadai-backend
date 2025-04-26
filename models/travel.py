from pydantic import BaseModel
from datetime import date
from typing import List, Optional, Dict, Any
from enum import Enum

class TravelPreference(str, Enum):
    CULTURE = "culture"
    RELAXATION = "relaxation"
    ADVENTURE = "adventure"
    FOOD = "food"
    NATURE = "nature"
    NIGHTLIFE = "nightlife"
    LUXURY = "luxury"
    BUDGET = "budget"
    FAMILY = "family"
    SHOPPING = "shopping"
    BEACH = "beach"
    MOUNTAIN = "mountain"

class TravelRequest(BaseModel):
    origin: str
    destination: str
    depart_date: date
    return_date: date
    duration: int
    budget: float
    preferences: List[str] = []
    adults: int = 1

class FlightOption(BaseModel):
    airline: str
    price: float
    origin: str
    destination: str
    depart_date: date
    return_date: date
    flight_number: Optional[str] = None
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None

class HotelOption(BaseModel):
    name: str
    price_per_night: float
    stars: float
    city: str
    address: Optional[str] = None
    amenities: Optional[List[str]] = None
    hotel_id: Optional[str] = None
    chain_code: Optional[str] = None
    distance: Optional[float] = None

class PointOfInterest(BaseModel):
    name: str
    category: str
    rating: float = 4.0
    address: Optional[str] = None
    description: Optional[str] = None
    price_level: Optional[int] = None
    image_url: Optional[str] = None

class ItineraryDayActivity(BaseModel):
    day: int
    date: date
    description: str
    morning: Optional[str] = None
    afternoon: Optional[str] = None
    evening: Optional[str] = None
    accommodation: Optional[str] = None

class Itinerary(BaseModel):
    request_id: str
    travel_request: TravelRequest
    selected_flight: Optional[FlightOption] = None
    selected_hotel: Optional[HotelOption] = None
    points_of_interest: List[PointOfInterest] = []
    daily_plan: List[ItineraryDayActivity] = []
    summary: str
    total_cost: float
    available_flights: List[FlightOption] = []
    available_hotels: List[HotelOption] = []
    raw_text: str = ""