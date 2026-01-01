from chatterbox.tts_turbo import ChatterboxTurboTTS
import torch

try:
    print("Loading TTS...")
    tts = ChatterboxTurboTTS.from_pretrained(device="cpu")
    print("\n--- METHODS ---")
    print(dir(tts))
    
    print("\n--- GENERATE DOCS ---")
    print(help(tts.generate))
    
except Exception as e:
    print(f"Error: {e}")
