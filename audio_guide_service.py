import os
import logging
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from config import settings

from openai import OpenAI
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Create audio directory
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
logger.info(f"Audio directory: {AUDIO_DIR}")

# Request model
class AudioGuideRequest(BaseModel):
    poiName: str
    poiCategory: str = ""
    poiDescription: str = ""

# Initialize TTS model
tts_model = None

def get_tts_model():
    """Get or initialize the TTS model"""
    global tts_model
    if tts_model is None:
        try:
            from TTS.api import TTS
            # Initialize the TTS model once and reuse it for performance
            tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            logger.info("TTS model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {str(e)}")
            raise HTTPException(status_code=500, detail="Could not initialize text-to-speech service")
    return tts_model

@router.post("/api/audio-guide")
async def generate_audio_guide(request: AudioGuideRequest):
    """
    Generate an audio guide for a point of interest using TTS.
    """
    try:
        logger.info(f"Audio request received for POI: {request.poiName}")
        
        # Generate prompt text for the audio guide
        prompt_text = generate_audio_prompt(
            request.poiName, 
            request.poiCategory, 
            request.poiDescription
        )
        
        # Generate a unique filename to avoid conflicts
        filename = f"guide_{uuid.uuid4()}.wav"
        file_path = os.path.join(AUDIO_DIR, filename)
        
        # Generate the audio file using TTS
        try:
            # Get the TTS model
            tts = get_tts_model()
            
            # Generate audio file
            tts.tts_to_file(
                text=prompt_text,
                file_path=file_path
            )
            logger.info(f"Generated audio file at {file_path}")
            
        except ImportError as e:
            logger.error(f"TTS module not available: {str(e)}")
            # Fallback to a sample file if TTS is not available
            return fallback_audio_response(request.poiName)
        except Exception as e:
            logger.error(f"Error generating audio with TTS: {str(e)}")
            return fallback_audio_response(request.poiName)
        
        # Return the URL to the generated audio file
        logger.info(f"Returning audio URL: /static/audio/{filename}")
        return {
            "audioUrl": f"/static/audio/{filename}",
            "text": prompt_text
        }
    except Exception as e:
        logger.error(f"Error generating audio guide: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def fallback_audio_response(poi_name):
    """Provide a fallback response when TTS fails"""
    sample_path = os.path.join(AUDIO_DIR, "sample.mp3")
    
    # Check if sample file exists, create a placeholder if not
    if not os.path.exists(sample_path):
        try:
            # Create a minimal valid MP3 file (just a few bytes)
            with open(sample_path, 'wb') as f:
                f.write(b'\xFF\xFB\x90\x44\x00')
            logger.info(f"Created placeholder MP3 file at {sample_path}")
        except Exception as e:
            logger.error(f"Failed to create sample file: {str(e)}")
    
    logger.info(f"Using fallback audio: /static/audio/sample.mp3")
    return {
        "audioUrl": "/static/audio/sample.mp3",
        "text": f"Welcome to {poi_name}. This is a sample audio guide (TTS unavailable)."
    }

def generate_audio_prompt(name, category, description):
    """
    Generate a natural-sounding prompt for the audio guide
    based on the POI information
    """
    # Default description if none provided
    if not description or description.strip() == "":
        description = f"a notable attraction in the area"
    
    # Clean category
    category_text = ""
    if category and category.strip() != "":
        # Format the category for natural language
        category = category.lower().replace("_", " ")
        category_text = f"This {category} spot "
    
    # Write a prompt for the audio guide
    prompt = f"You are a tour guide creating an audio experience for someone currently inside the point of interest (POI) named '{name}'. The POI is categorized as '{category}' and described as '{description}'. Generate a short, natural-sounding audio script (under 100 words) that makes the listener feel like they are experiencing something amazing and encourages them to explore further. The script should be in plain text, suitable for a voiceover, without any bracketed annotations or sound cues. Start with an engaging opening that evokes a sense of wonder."

    # Generate a guided tour script using openai
    response = OpenAI().chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content