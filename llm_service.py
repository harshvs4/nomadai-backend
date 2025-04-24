import json
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

import openai
from openai import OpenAI
from fastapi import HTTPException, status

from config import settings
from models.travel import (
    TravelRequest, FlightOption, HotelOption, 
    PointOfInterest, Itinerary, ItineraryDayActivity
)
from amadeus_service import AmadeusService
from google_places_service import GooglePlacesService

logger = logging.getLogger(__name__)

class LLMPlanningService:
    """
    Service for using LLMs to generate travel itineraries based on
    flight, hotel, and point of interest data.
    """
    
    SYSTEM_PROMPT = """
    You are NomadAI, an intelligent travel planning assistant that creates personalized travel itineraries.
    Your task is to create a detailed day-by-day travel plan based on the provided flight, hotel, and points of interest data.
    
    Guidelines:
    1. Create a practical, coherent, and well-structured itinerary that respects the user's budget and preferences
    2. Distribute points of interest across days in a logical way, considering location and opening times
    3. Include specific flight and hotel recommendations from the provided options
    4. Add practical details like transportation between attractions
    5. Make the itinerary realistic in terms of timing and distances
    6. Include estimated costs for activities when possible
    7. Format the output in clear, well-organized markdown
    8. Make sure the total cost (flight + hotel + activities) stays within the user's budget
    
    Your response should include:
    - A brief introduction summarizing the trip
    - A suggested flight and hotel with prices
    - A day-by-day breakdown of activities (morning, afternoon, evening)
    - An estimated total cost breakdown
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
    
    async def generate_itinerary(self, 
                               travel_request: TravelRequest,
                               amadeus_service: AmadeusService,
                               google_places_service: GooglePlacesService,
                               cache_service) -> Itinerary:
        """
        Generate a complete travel itinerary using the LLM.
        This is an async wrapper around the create_itinerary method.
        """
        # Get flight options
        flights = amadeus_service.get_flight_offers(
            travel_request.origin,
            travel_request.destination,
            travel_request.depart_date,
            travel_request.return_date
        )
        
        if not flights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No flights found from {travel_request.origin} to {travel_request.destination}"
            )
        
        # Get hotel options using the correct method
        hotels_data = amadeus_service.search_hotels_by_city(travel_request.destination)
        if not hotels_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No hotels found in {travel_request.destination}"
            )
        
        # Convert hotel data to HotelOption objects
        hotels = []
        for hotel in hotels_data[:10]:  # Limit to 10 hotels
            # Generate a price based on stars or a default value
            import random
            hotel_stars = float(hotel.get("rating", 3))
            price_per_night = 100 + (hotel_stars * 50) * random.uniform(0.8, 1.2)
            
            hotels.append(HotelOption(
                name=hotel.get("name", "Unknown Hotel"),
                price_per_night=price_per_night,
                stars=hotel_stars,
                address=hotel.get("address", {}).get("countryCode", ""),
                amenities=["Wi-Fi", "Room Service", "Air Conditioning"],
                city=travel_request.destination,
                hotel_id=hotel.get("hotelId", "")
            ))
        
        # Get points of interest
        pois = google_places_service.get_points_of_interest(
            travel_request.destination,
            travel_request.preferences
        )
        
        # Generate the itinerary
        return self.create_itinerary(travel_request, flights, hotels, pois)
    
    def create_itinerary(self, 
                         travel_request: TravelRequest,
                         flights: List[FlightOption],
                         hotels: List[HotelOption],
                         pois: List[PointOfInterest]) -> Itinerary:
        """
        Generate a complete travel itinerary using the LLM.
        """
        # Prepare the context for the LLM
        context = self._prepare_context(travel_request, flights, hotels, pois)
        
        # Generate the itinerary text using the LLM
        itinerary_text = self._generate_itinerary_text(travel_request, context)
        
        # Parse the generated text into structured data
        itinerary = self._parse_itinerary(travel_request, itinerary_text, flights, hotels, pois)
        
        # Important: Store the available flights and hotels in the itinerary
        # to allow for user selection later
        itinerary.available_flights = flights[:5] if len(flights) > 5 else flights
        itinerary.available_hotels = hotels[:5] if len(hotels) > 5 else hotels
        
        return itinerary
        
    def _prepare_context(self, 
                        travel_request: TravelRequest,
                        flights: List[FlightOption],
                        hotels: List[HotelOption],
                        pois: List[PointOfInterest]) -> Dict[str, Any]:
        """
        Prepare a structured context dictionary for the LLM prompt.
        """
        # Clean and prepare the data for serialization
        flight_data = []
        # Use only the top 5 flights for the LLM context to keep it manageable
        for f in flights[:5]:
            flight_data.append({
                "airline": f.airline,
                "price": f.price,
                "departure_time": f.departure_time if hasattr(f, 'departure_time') else None,
                "arrival_time": f.arrival_time if hasattr(f, 'arrival_time') else None,
                "flight_number": f.flight_number if hasattr(f, 'flight_number') else None
            })
        
        hotel_data = []
        # Use only the top 5 hotels for the LLM context to keep it manageable
        for h in hotels[:5]:
            hotel_data.append({
                "name": h.name,
                "price_per_night": h.price_per_night,
                "stars": h.stars,
                "address": h.address if hasattr(h, 'address') else None,
                "amenities": h.amenities if hasattr(h, 'amenities') else None
            })
        
        poi_data = []
        for p in pois:
            poi_dict = {
                "name": p.name,
                "category": p.category,
                "rating": p.rating,
                "address": p.address
            }
            
            # Add optional fields if they exist
            if hasattr(p, 'description') and p.description:
                poi_dict["description"] = p.description
                
            if hasattr(p, 'price_level') and p.price_level is not None:
                poi_dict["price_level"] = p.price_level
                
            if hasattr(p, 'image_url') and p.image_url:
                poi_dict["image_url"] = p.image_url
                
            poi_data.append(poi_dict)
        
        # Create the context dictionary
        context = {
            "trip_details": {
                "origin": travel_request.origin,
                "destination": travel_request.destination,
                "duration_days": travel_request.duration,
                "start_date": travel_request.depart_date.isoformat(),
                "end_date": travel_request.return_date.isoformat(),
                "budget": travel_request.budget,
                "preferences": travel_request.preferences
            },
            "flights": flight_data,
            "hotels": hotel_data,
            "points_of_interest": poi_data
        }
        
        return context
    
    def _generate_itinerary_text(self, travel_request: TravelRequest, context: Dict[str, Any]) -> str:
        """
        Generate itinerary text using the OpenAI API.
        """
        try:
            # Create a user message with the planning request and context
            trip_desc = (f"Please plan a {travel_request.duration}-day trip from {travel_request.origin} "
                        f"to {travel_request.destination} with a budget of SGD {travel_request.budget}.")
            
            preferences = "no specific preferences"
            if travel_request.preferences:
                preferences = ", ".join(travel_request.preferences)
            
            user_message = f"{trip_desc}\n\nThe traveler has indicated the following preferences: {preferences}.\n\n"
            user_message += f"Data:\n{json.dumps(context, indent=2, default=str)}"
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.5,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Extract and return the generated text
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating itinerary with LLM: {str(e)}")
            # Provide a fallback simple itinerary
            return self._generate_fallback_itinerary(travel_request)
    
    def _generate_fallback_itinerary(self, travel_request: TravelRequest) -> str:
        """Generate a simple fallback itinerary if the LLM call fails."""
        duration = travel_request.duration
        destination = travel_request.destination
        
        fallback = f"# Trip to {destination}\n\n"
        fallback += f"## Overview\n"
        fallback += f"A {duration}-day trip to {destination} from {travel_request.origin}.\n\n"
        
        fallback += f"## Suggested Flight\n"
        fallback += f"Economy class flight from {travel_request.origin} to {destination}.\n\n"
        
        fallback += f"## Suggested Accommodation\n"
        fallback += f"Standard hotel in {destination} city center.\n\n"
        
        fallback += f"## Day-by-Day Itinerary\n"
        
        # Generate a simple day-by-day itinerary
        for day in range(1, duration + 1):
            fallback += f"### Day {day}\n"
            fallback += f"- Morning: Breakfast at hotel, explore local area\n"
            fallback += f"- Afternoon: Visit main tourist attractions\n"
            fallback += f"- Evening: Dinner at local restaurant\n\n"
        
        fallback += f"## Estimated Budget\n"
        fallback += f"Total estimated cost: SGD {travel_request.budget * 0.9:.2f}\n"
        
        return fallback
    
    def _parse_itinerary(self, 
                    travel_request: TravelRequest, 
                    itinerary_text: str,
                    flights: List[FlightOption],
                    hotels: List[HotelOption],
                    pois: List[PointOfInterest]) -> Itinerary:
        """
        Parse the generated itinerary text into a structured Itinerary object.
        Calculate the actual trip cost and ensure it stays within budget.
        """
        # Select default flight and hotel (the first ones in their respective lists)
        # Sort flights by price to get the cheapest option first
        sorted_flights = sorted(flights, key=lambda x: x.price) if flights else []
        sorted_hotels = sorted(hotels, key=lambda x: x.price_per_night) if hotels else []
        
        selected_flight = sorted_flights[0] if sorted_flights else None
        selected_hotel = sorted_hotels[0] if sorted_hotels else None
        
        # Try to find mentioned flight in the itinerary text, but only if it fits in budget
        for flight in flights:
            flight_identifier = f"{flight.airline} {flight.flight_number}" if hasattr(flight, 'flight_number') else flight.airline
            if flight_identifier in itinerary_text:
                # Check if this flight would fit in budget
                if flight.price <= travel_request.budget * 0.6:  # Allow up to 60% of budget for flight
                    selected_flight = flight
                    break
        
        # Try to find mentioned hotel in the itinerary text, but only if it fits in budget
        nights = max(1, travel_request.duration - 1)
        remaining_budget = travel_request.budget - (selected_flight.price if selected_flight else 0)
        max_hotel_per_night = remaining_budget / nights * 0.7  # Allow up to 70% of remaining budget for hotel
        
        for hotel in hotels:
            if hotel.name in itinerary_text:
                # Check if this hotel would fit in budget
                if hotel.price_per_night <= max_hotel_per_night:
                    selected_hotel = hotel
                    break
        
        # Create a day-by-day plan from the markdown text
        daily_plan = []
        days = travel_request.duration
        
        # Split by days and extract content
        for day in range(1, days + 1):
            # Look for day headers in the markdown
            day_markers = [
                f"### Day {day}",
                f"## Day {day}",
                f"Day {day}:",
                f"Day {day} -"
            ]
            
            day_content = ""
            for marker in day_markers:
                if marker in itinerary_text:
                    parts = itinerary_text.split(marker, 1)
                    if len(parts) > 1:
                        next_day_idx = float('inf')
                        for next_day in range(day + 1, days + 1):
                            for next_marker in day_markers:
                                next_marker = next_marker.replace(str(day), str(next_day))
                                if next_marker in parts[1]:
                                    marker_idx = parts[1].find(next_marker)
                                    if marker_idx < next_day_idx:
                                        next_day_idx = marker_idx
                        
                        if next_day_idx < float('inf'):
                            day_content = parts[1][:next_day_idx]
                        else:
                            day_content = parts[1]
                        break
            
            # If we found content for this day
            if day_content:
                # Extract morning, afternoon, evening activities more thoroughly
                morning = afternoon = evening = None
                
                # Morning
                morning_markers = ["Morning:", "**Morning:**", "#### Morning", "### Morning"]
                for marker in morning_markers:
                    if marker in day_content:
                        morning_parts = day_content.split(marker, 1)
                        if len(morning_parts) > 1:
                            # Try to find the end of the morning section
                            afternoon_marker_idx = float('inf')
                            for afternoon_marker in ["Afternoon:", "**Afternoon:**", "#### Afternoon", "### Afternoon"]:
                                if afternoon_marker in morning_parts[1]:
                                    marker_idx = morning_parts[1].find(afternoon_marker)
                                    if marker_idx < afternoon_marker_idx:
                                        afternoon_marker_idx = marker_idx
                            
                            if afternoon_marker_idx < float('inf'):
                                morning = morning_parts[1][:afternoon_marker_idx].strip()
                            else:
                                # If no afternoon marker, look for evening marker
                                evening_marker_idx = float('inf')
                                for evening_marker in ["Evening:", "**Evening:**", "#### Evening", "### Evening"]:
                                    if evening_marker in morning_parts[1]:
                                        marker_idx = morning_parts[1].find(evening_marker)
                                        if marker_idx < evening_marker_idx:
                                            evening_marker_idx = marker_idx
                                
                                if evening_marker_idx < float('inf'):
                                    morning = morning_parts[1][:evening_marker_idx].strip()
                                else:
                                    # If no section markers, take the first paragraph
                                    if "\n\n" in morning_parts[1]:
                                        morning = morning_parts[1].split("\n\n", 1)[0].strip()
                                    else:
                                        morning = morning_parts[1].split("\n", 1)[0].strip()
                        break
                
                # Afternoon
                afternoon_markers = ["Afternoon:", "**Afternoon:**", "#### Afternoon", "### Afternoon"]
                for marker in afternoon_markers:
                    if marker in day_content:
                        afternoon_parts = day_content.split(marker, 1)
                        if len(afternoon_parts) > 1:
                            # Try to find the end of the afternoon section
                            evening_marker_idx = float('inf')
                            for evening_marker in ["Evening:", "**Evening:**", "#### Evening", "### Evening"]:
                                if evening_marker in afternoon_parts[1]:
                                    marker_idx = afternoon_parts[1].find(evening_marker)
                                    if marker_idx < evening_marker_idx:
                                        evening_marker_idx = marker_idx
                            
                            if evening_marker_idx < float('inf'):
                                afternoon = afternoon_parts[1][:evening_marker_idx].strip()
                            else:
                                # If no evening marker, take the first paragraph
                                if "\n\n" in afternoon_parts[1]:
                                    afternoon = afternoon_parts[1].split("\n\n", 1)[0].strip()
                                else:
                                    afternoon = afternoon_parts[1].split("\n", 1)[0].strip()
                        break
                
                # Evening
                evening_markers = ["Evening:", "**Evening:**", "#### Evening", "### Evening"]
                for marker in evening_markers:
                    if marker in day_content:
                        evening_parts = day_content.split(marker, 1)
                        if len(evening_parts) > 1:
                            # Take until the next major section or the end
                            if "\n\n" in evening_parts[1]:
                                evening = evening_parts[1].split("\n\n", 1)[0].strip()
                            else:
                                evening = evening_parts[1].strip()
                        break
                
                # Calculate the date for this day
                day_date = travel_request.depart_date
                if day > 1:
                    from datetime import timedelta
                    day_date = travel_request.depart_date + timedelta(days=day-1)
                
                # Ensure the accommodation is consistent with the selected hotel
                daily_plan.append(ItineraryDayActivity(
                    day=day,
                    date=day_date,
                    description=day_content.strip(),
                    morning=morning,
                    afternoon=afternoon,
                    evening=evening,
                    accommodation=selected_hotel.name if selected_hotel else None
                ))
        
        # Calculate actual cost of the trip
        # Start with flight cost
        actual_cost = selected_flight.price if selected_flight else 0
        
        # Add hotel cost (price per night * number of nights)
        if selected_hotel:
            # Number of nights is duration - 1 (unless it's a day trip)
            nights = max(1, travel_request.duration - 1)
            actual_cost += selected_hotel.price_per_night * nights
        
        # Try to extract activity costs from the itinerary text
        # Look for cost information in the itinerary text
        import re
        
        # First try to find a total cost explicitly mentioned in the text
        cost_patterns = [
            r'Total cost:\s*SGD\s*([\d,]+(\.\d+)?)',
            r'Total Cost:\s*SGD\s*([\d,]+(\.\d+)?)',
            r'estimated total cost.*?SGD\s*([\d,]+(\.\d+)?)',
            r'total budget.*?SGD\s*([\d,]+(\.\d+)?)',
            r'estimated cost.*?SGD\s*([\d,]+(\.\d+)?)'
        ]
        
        extracted_cost = None
        for pattern in cost_patterns:
            cost_match = re.search(pattern, itinerary_text, re.IGNORECASE)
            if cost_match:
                try:
                    extracted_cost = float(cost_match.group(1).replace(',', ''))
                    # Only use the extracted cost if it's within budget
                    if extracted_cost <= travel_request.budget:
                        actual_cost = extracted_cost
                    break
                except (ValueError, IndexError):
                    pass
        
        # If we couldn't extract a total cost, look for activity costs in the text
        if not extracted_cost:
            # This pattern looks for SGD followed by numbers 
            activity_cost_pattern = r'SGD\s*([\d,]+(\.\d+)?)'
            activity_costs = re.findall(activity_cost_pattern, itinerary_text)
            
            # Add any activity costs found (excluding the ones that might be for flights/hotels)
            activity_total = 0
            for cost_match in activity_costs:
                try:
                    cost = float(cost_match[0].replace(',', ''))
                    # Skip if the cost is likely a flight or hotel cost
                    # Only add costs below 500 to avoid counting flights/total costs
                    if cost < 500 and cost > 0:
                        activity_total += cost
                except (ValueError, IndexError):
                    continue
            
            # Add activity costs to the total, but cap at remaining budget
            remaining_budget = travel_request.budget - actual_cost
            actual_cost += min(activity_total, remaining_budget)
        
        # ENSURE total cost never exceeds budget
        if actual_cost > travel_request.budget:
            # Cap the cost at the budget
            actual_cost = travel_request.budget
            
        # IMPORTANT: If the final cost is too low (less than 50% of budget), 
        # set it to a reasonable percentage of the budget to make it realistic
        if actual_cost < travel_request.budget * 0.5:
            actual_cost = travel_request.budget * 0.8  # 80% of budget
            
        # Extract a summary from the beginning of the itinerary
        summary = ""
        if "# " in itinerary_text:
            # Try to get the content after the first heading
            parts = itinerary_text.split("# ", 1)
            if len(parts) > 1:
                if "\n\n" in parts[1]:
                    summary = parts[1].split("\n\n", 1)[0].strip()
                else:
                    summary = parts[1].strip()
        
        if not summary and "\n\n" in itinerary_text:
            summary = itinerary_text.split("\n\n", 1)[0].strip()
        
        if not summary:
            summary = itinerary_text[:200].strip()
        
        # Create and return the Itinerary object
        return Itinerary(
            request_id=str(uuid.uuid4()),
            travel_request=travel_request,
            selected_flight=selected_flight,
            selected_hotel=selected_hotel,
            points_of_interest=pois,
            daily_plan=daily_plan,
            summary=summary,
            total_cost=actual_cost,  # Use our budget-constrained cost
            available_flights=flights[:5] if len(flights) > 5 else flights,
            available_hotels=hotels[:5] if len(hotels) > 5 else hotels
        )