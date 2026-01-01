from chatterbox.tts_turbo import ChatterboxTurboTTS
import inspect

try:
    print("Loading TTS...")
    tts = ChatterboxTurboTTS.from_pretrained(device="cpu")
    
    print("\n--- SIGNATURES ---")
    try:
        print(f"prepare_conditionals: {inspect.signature(tts.prepare_conditionals)}")
    except:
        print("prepare_conditionals: (Could not get signature)")
        
    try:
        print(f"generate: {inspect.signature(tts.generate)}")
    except:
        print("generate: (Could not get signature)")
        
except Exception as e:
    print(f"Error: {e}")
