import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import date
from typing import List, Optional
import uvicorn

from amadeus_service import AmadeusService
from google_places_service import GooglePlacesService
from llm_service import LLMPlanningService
from audio_guide_service import router as audio_guide_router

# Initialize services
amadeus_service = AmadeusService()
google_places_service = GooglePlacesService()
llm_service = LLMPlanningService()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include the audio guide router
app.include_router(audio_guide_router)


# Models
class TravelRequestModel(BaseModel):
    origin: str
    destination: str
    depart_date: date
    return_date: date
    duration: int
    budget: float
    preferences: List[str] = []
    adults: int = 1

class ChatMessageModel(BaseModel):
    request_id: str
    message: str

class ItineraryUpdateModel(BaseModel):
    selected_flight: Optional[dict] = None
    selected_hotel: Optional[dict] = None
    daily_plan_updates: Optional[List[dict]] = None

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the Travel Planning API"}

@app.get("/api")
async def api_root():
    return {"message": "API is running"}

@app.get("/api/flights")
async def get_flights(
    origin: str, 
    destination: str, 
    depart_date: date, 
    return_date: date, 
    adults: int = 1
):
    try:
        flights = amadeus_service.get_flight_offers(
            origin, destination, depart_date, return_date, adults
        )
        return flights
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hotels")
async def get_hotels(
    destination: str, 
    checkin: date, 
    checkout: date, 
    adults: int = 1, 
    rooms: int = 1
):
    try:
        hotels = amadeus_service.get_hotel_offers(
            destination, checkin, checkout, adults, rooms
        )
        return hotels
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/points-of-interest")
async def get_points_of_interest(
    destination: str, 
    preferences: Optional[str] = None
):
    try:
        prefs_list = preferences.split(',') if preferences else []
        pois = google_places_service.get_points_of_interest(
            destination, prefs_list
        )
        return pois
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/itinerary/generate")
async def generate_itinerary(request: TravelRequestModel):
    try:
        from models.travel import TravelRequest
        
        # Convert Pydantic model to TravelRequest
        travel_request = TravelRequest(
            origin=request.origin,
            destination=request.destination,
            depart_date=request.depart_date,
            return_date=request.return_date,
            duration=request.duration,
            budget=request.budget,
            preferences=request.preferences,
            adults=request.adults
        )
        
        # Create cache service stub (or implement if needed)
        cache_service = None
        
        # Generate itinerary
        itinerary = await llm_service.generate_itinerary(
            travel_request, amadeus_service, google_places_service, cache_service
        )
        
        return itinerary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(message: ChatMessageModel):
    try:
        # Process the message with LLM
        response = await llm_service.sendChatMessage(
            itinerary_id=message.request_id,
            message=message.message
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.patch("/api/itinerary/{request_id}")
async def update_itinerary(request_id: str, updates: ItineraryUpdateModel):
    try:
        # Update the itinerary
        # Note: You'll need to implement this in llm_service.py
        # or adapt the existing methods
        
        # For now, we'll just return a success message
        # In a real implementation, you'd fetch the itinerary,
        # apply the updates, and return the updated itinerary
        return {
            "message": f"Itinerary {request_id} updated successfully",
            "updates": updates.dict(exclude_none=True)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)