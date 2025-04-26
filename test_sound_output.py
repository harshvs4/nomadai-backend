from TTS.api import TTS

# choose a pretrained model (e.g., Tacotron2 + HiFi-GAN on LJSpeech)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# generate speech and save as WAV
tts.tts_to_file(
    text="Hello, Harsh! This is a demo of text to speech.",
    file_path="output.wav"
)
