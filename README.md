# NomadAI Backend

A FastAPI-based backend service for intelligent travel planning, integrating flight bookings, hotel reservations, and personalized itinerary generation using AI.

## Features

- **Flight Search**: Integration with Amadeus API for real-time flight availability and pricing
- **Hotel Booking**: Comprehensive hotel search and booking capabilities
- **Points of Interest**: Google Places integration for discovering local attractions
- **AI-Powered Planning**: LLM-based itinerary generation and personalization
- **Audio Guide**: AI-generated audio guides for points of interest using TTS technology
- **RESTful API**: Clean, documented endpoints for easy integration

## Tech Stack

- **Framework**: FastAPI
- **Language**: Python 3.8+
- **APIs**:
  - Amadeus API (Flights & Hotels)
  - Google Places API
  - OpenAI API (for LLM services)
- **AI Models**:
  - OpenAI GPT-4 (for itinerary generation and chat)
  - TTS (Text-to-Speech) models:
    - Tacotron2-DDC (for audio guide generation)
- **Database**: (To be implemented)

## Project Structure

```
backend/
├── main.py                           # FastAPI application entry point
├── config.py                         # Configuration and environment settings
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables
├── amadeus_service.py                # Service for flight and hotel data
├── google_places_service.py          # Service for points of interest
├── llm_service.py                    # LLM-based planning service
└── models/
    └── travel.py                     # Data models for travel planning
```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/harshvs4/nomadai-backend.git
   cd nomadai-backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys and configuration

5. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

- `GET /api/flights` - Search for flights
- `GET /api/hotels` - Search for hotels
- `GET /api/points-of-interest` - Get local attractions
- `POST /api/itinerary/generate` - Generate personalized itinerary
- `POST /api/chat` - Interactive chat for itinerary modifications
- `POST /api/audio-guide` - Generate audio guide for points of interest

## Environment Variables

Create a `.env` file with the following variables:
```
AMADEUS_API_KEY=your_amadeus_api_key
AMADEUS_API_SECRET=your_amadeus_api_secret
GOOGLE_PLACES_API_KEY=your_google_places_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- FastAPI for the amazing web framework
- Amadeus for Travel APIs
- Google Places API
- OpenAI for LLM capabilities 

## Team

- Gudur Venkata Rajeshwari (A0297977W)
- Harsh Sharma (A0295906N)
- Shivika Mathur (A0298106Y)
- Soumya Haridas (A0296635N)
- Vijit Daroch (A0296998R)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 