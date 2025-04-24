import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY", "demo")
    AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET", "demo")
    GOOGLE_PLACES_KEY = os.getenv("GOOGLE_PLACES_KEY", "demo")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "demo")
    
    # Service settings
    AMADEUS_TEST_MODE = os.getenv("AMADEUS_TEST_MODE", "True").lower() == "true"
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-1106-preview")

settings = Settings()