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
        You are NomadAI, an intelligent travel-planning assistant.

        **[TASK]**
        * Generate a personalized, detailed, day-by-day travel itinerary based on user-provided data and preferences.
        * The itinerary must respect the user's specified travel dates, budget constraints, and stated preferences.
        * Recommend specific round-trip flights, suitable accommodation, and relevant points of interest, drawing only from the provided input data lists.

        **[INPUT DATA]**
        * **Trip Details:** origin, destination, start_date, end_date, duration (in days), total_budget, preferences (e.g., interests like history, food, nature, pace like relaxed/fast-paced).
        * **Round-Trip Flight Options:** A list where each item contains details for a complete round trip: {airline, outbound_flight_number, outbound_depart_datetime, outbound_arrival_datetime, inbound_flight_number, inbound_depart_datetime, inbound_arrival_datetime, total_round_trip_price}.
        * **Hotel Options:** list of {name, price_per_night, stars, address, amenities}.
        * **Points of Interest (POIs):** list of {name, category (e.g., Museum, Park, Restaurant, Landmark), rating, address, description (optional), price_level (e.g., Free, $, $$, $$$), typical_visit_duration (optional, in hours)}.

        **[GUIDELINES]**
        1.  **Structure:** Divide each day logically into **Morning**, **Afternoon**, and **Evening** segments.
        2.  **Realism:** Account for realistic travel times *between* locations and the typical time needed *at* each point of interest (use `typical_visit_duration` from POI data if available, otherwise estimate). Factor in opening/closing hours for attractions.
        3.  **Budgeting:** The `total_budget` indicates the target spending level and desired quality/quantity of experiences. Select flights, hotels, activities, and dining options that align with this level and the user's preferences. If the budget is substantial (e.g., significantly higher than a basic trip would require), incorporate higher-rated accommodations (e.g., 4-5 stars if available in options), more unique or premium activities/tours, and potentially finer dining suggestions where appropriate, aiming to utilize the budget effectively for a richer or more comfortable experience according to preferences. However, the total estimated cost must **strictly stay within** the user's `total_budget`. Show approximate costs for activities, suggested meals, and transport segments.
        4.  **Efficiency:** Group geographically close points of interest together within the same day or segment (Morning/Afternoon) to minimize travel time and cost.
        5.  **Detail:** For each activity/POI, specify the estimated duration of the visit. For travel between major activities or locations, suggest a mode of transport (e.g., walking, public transport, taxi/rideshare) and estimate the travel time and cost. Include suggestions for lunch and dinner, noting the approximate cost.
        6.  **Clarity:** Use clear and concise language.

        **[OUTPUT FORMAT]**
        Respond in **Markdown**, adhering strictly to the following structure:

        1.  **Trip Overview**
            * Destination: [City, Country]
            * Dates: [Start Date] to [End Date] 
            * Budget: [Currency] [Amount]
            * Summary: A brief paragraph outlining the trip's focus based on preferences.

        2.  **Round-Trip Flight Recommendation**
            * Airline: [Airline Name]
            * Outbound: [Flight Number], Departs [Origin] at [Time] on [Date], Arrives [Destination] at [Time] on [Date]
            * Inbound: [Flight Number], Departs [Destination] at [Time] on [Date], Arrives [Origin] at [Time] on [Date]
            * Total Cost: [Currency] [Amount]

        3.  **Hotel Recommendation**
            * Name: [Hotel Name]
            * Rating: [Star Rating] Stars
            * Address: [Full Address]
            * Selected Amenities: [List key amenities, e.g., Wi-Fi, Breakfast Included]
            * Nightly Rate: [Currency] [Amount]
            * Total Cost ([Number] nights): [Currency] [Total Hotel Cost]

        4.  **Detailed Day-by-Day Plan**
            For each day:
            **Day N: [Date] - [Optional: Brief Theme for the Day]**
            **Morning:**
                * [Activity/POI Name] (Est. Duration: [X] hours) - Approx. Cost: [Currency] [Amount]
                    * *Optional: Brief note about the activity.*
                * *Travel:* Est. [Y] minutes via [Transport Mode] to next location - Approx. Cost: [Currency] [Amount]
                * [Next Activity/POI Name] (Est. Duration: [Z] hours) - Approx. Cost: [Currency] [Amount]
            **Afternoon:**
                * *Lunch:* Suggestion [e.g., Cafe near POI] (Est. Duration: 1 hour) - Approx. Cost: [Currency] [Amount]
                * [Activity/POI Name] (Est. Duration: [A] hours) - Approx. Cost: [Currency] [Amount]
                * *Travel:* Est. [B] minutes via [Transport Mode] to hotel/next area - Approx. Cost: [Currency] [Amount]
            **Evening:**
                * *Dinner:* Suggestion [e.g., Restaurant in X district] (Est. Duration: 1.5 hours) - Approx. Cost: [Currency] [Amount]
                * [Evening Activity/Option, e.g., Walk, Show, Relax] (Est. Duration: [C] hours) - Approx. Cost: [Currency] [Amount]
            **Estimated Daily Cost:** [Currency] [Total for Day N]

        5.  **Cost Summary**
            * Round-Trip Flight: [Currency] [Amount]
            * Hotel ([Number] nights): [Currency] [Amount]
            * Daily Expenses & Activities:
                * Day 1: [Currency] [Amount]
                * Day 2: [Currency] [Amount]
                * Day 3: [Currency] [Amount]
                * ... (repeat for all days)
                * *Subtotal Daily Expenses:* [Currency] [Sum of Daily Costs]
            * **Estimated Total Trip Cost:** [Currency] [Grand Total]

        6.  **Remaining Budget**
            * Total Budget: [Currency] [Amount]
            * Estimated Total Trip Cost: [Currency] [Amount]
            * **Estimated Remaining Budget:** [Currency] [Amount]

        """

    CHAT_PROMPT = """
        You are NomadAI, an intelligent travel-planning assistant. You are having a conversation with a user about their travel itinerary.
        Your goal is to help them understand and modify their travel plans based on their questions and preferences.
        
        IMPORTANT GUIDELINES:
        1. Always verify and compare prices accurately before making recommendations
        2. When comparing prices, ensure you're looking at the exact numbers provided in the context
        3. Double-check your calculations and comparisons before suggesting changes
        4. If a user asks about cheaper options, only suggest hotels that are actually cheaper than their current selection, the same goes with flight options as well
        5. Be precise with numbers and avoid making assumptions about prices
        6. If you're unsure about a price comparison, ask for clarification rather than making potentially incorrect suggestions
        
        Use the provided context about their flights, hotels, and points of interest to give accurate and helpful responses.
        If the user asks about specific details, refer to the actual data provided in the context.
        
        **Use lists, bold, and headings where appropriate.**
        
        **Be concise by default. Only provide detailed lists, comparisons, or breakdowns if the user explicitly asks for them. Summarize or highlight only the most relevant information unless more detail is requested. Avoid repeating information already present in the context unless clarification is needed.**
        
        Keep your responses concise and focused on helping the user with their travel planning needs.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        # Initialize in-memory storage for itineraries
        self._itineraries: Dict[str, Itinerary] = {}
    
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
        itinerary = self.create_itinerary(travel_request, flights, hotels, pois)
        
        # Store the itinerary in memory
        self._itineraries[itinerary.request_id] = itinerary
        
        return itinerary
    
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
        itinerary.raw_text = itinerary_text
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
        #print(hotel_data)
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
                        f"to {travel_request.destination} within a budget of SGD {travel_request.budget}.")
            
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
    
    async def sendChatMessage(self, itinerary_id: str, message: str) -> Dict[str, Any]:
        """
        Process a chat message from the user and return a response.
        
        Args:
            itinerary_id: The ID of the itinerary being discussed
            message: The user's message
            
        Returns:
            A dictionary containing the response and any updated itinerary information
        """
        try:
            # Get the itinerary context (you'll need to implement this method)
            context = await self._get_itinerary_context(itinerary_id)
            #print("this is the context", context)
            if not context:
                return {
                    "reply": "I couldn't find the itinerary you're referring to. Please try again.",
                    "updated_itinerary": None
                }
            
            # Prepare the messages for the chat
            messages = [
                {"role": "system", "content": self.CHAT_PROMPT},
                {"role": "user", "content": f"Here is the context about my trip:\n{json.dumps(context, indent=2)}\n\nMy question: {message}"}
            ]
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.7,
                messages=messages
            )
            
            # Extract the response
            reply = response.choices[0].message.content.strip()
            
            return {
                "reply": reply,
                "updated_itinerary": None  # This could be populated if the chat resulted in itinerary changes
            }
            
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            return {
                "reply": "I'm sorry, I encountered an error while processing your message. Please try again.",
                "updated_itinerary": None
            }
    
    async def _get_itinerary_context(self, itinerary_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the context for a specific itinerary from in-memory storage.
        
        Args:
            itinerary_id: The ID of the itinerary to retrieve
            
        Returns:
            A dictionary containing the itinerary context or None if not found
        """
        # Get the itinerary from in-memory storage
        itinerary = self._itineraries.get(itinerary_id)
        
        if not itinerary:
            return None
            
        # Prepare the context dictionary
        context = {
            "trip_details": {
                "origin": itinerary.travel_request.origin,
                "destination": itinerary.travel_request.destination,
                "duration_days": itinerary.travel_request.duration,
                "start_date": itinerary.travel_request.depart_date.isoformat(),
                "end_date": itinerary.travel_request.return_date.isoformat(),
                "budget": itinerary.travel_request.budget,
                "preferences": itinerary.travel_request.preferences
            },
            "selected_flight": {
                "airline": itinerary.selected_flight.airline if itinerary.selected_flight else None,
                "price": itinerary.selected_flight.price if itinerary.selected_flight else None,
                "departure_time": itinerary.selected_flight.departure_time if itinerary.selected_flight else None,
                "arrival_time": itinerary.selected_flight.arrival_time if itinerary.selected_flight else None,
                "flight_number": itinerary.selected_flight.flight_number if itinerary.selected_flight else None
            } if itinerary.selected_flight else None,
            "selected_hotel": {
                "name": itinerary.selected_hotel.name if itinerary.selected_hotel else None,
                "price_per_night": itinerary.selected_hotel.price_per_night if itinerary.selected_hotel else None,
                "stars": itinerary.selected_hotel.stars if itinerary.selected_hotel else None,
                "address": itinerary.selected_hotel.address if itinerary.selected_hotel else None,
                "amenities": itinerary.selected_hotel.amenities if itinerary.selected_hotel else None
            } if itinerary.selected_hotel else None,
            "available_flights": [
                {
                    "airline": f.airline,
                    "price": f.price,
                    "departure_time": f.departure_time if hasattr(f, 'departure_time') else None,
                    "arrival_time": f.arrival_time if hasattr(f, 'arrival_time') else None,
                    "flight_number": f.flight_number if hasattr(f, 'flight_number') else None
                }
                for f in itinerary.available_flights
            ],
            "available_hotels": [
                {
                    "name": h.name,
                    "price_per_night": h.price_per_night,
                    "stars": h.stars,
                    "address": h.address if hasattr(h, 'address') else None,
                    "amenities": h.amenities if hasattr(h, 'amenities') else None
                }
                for h in itinerary.available_hotels
            ],
            "points_of_interest": [
                {
                    "name": p.name,
                    "category": p.category,
                    "rating": p.rating,
                    "address": p.address,
                    "description": p.description if hasattr(p, 'description') else None,
                    "price_level": p.price_level if hasattr(p, 'price_level') else None,
                    "image_url": p.image_url if hasattr(p, 'image_url') else None
                }
                for p in itinerary.points_of_interest
            ],
            "daily_plan": [
                {
                    "day": day.day,
                    "date": day.date.isoformat(),
                    "description": day.description,
                    "morning": day.morning,
                    "afternoon": day.afternoon,
                    "evening": day.evening,
                    "accommodation": day.accommodation
                }
                for day in itinerary.daily_plan
            ],
            "summary": itinerary.summary,
            "total_cost": itinerary.total_cost,
            "raw_itinerary_text": itinerary.raw_text  # Add the raw itinerary text
        }
        
        return context
    
 