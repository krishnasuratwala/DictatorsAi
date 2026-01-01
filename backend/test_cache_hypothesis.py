from chatterbox.tts_turbo import ChatterboxTurboTTS
import torch
import os

AUDIO_PROMPT = "hitler_calm_voice.mp3"

try:
    print("Loading TTS...")
    tts = ChatterboxTurboTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(AUDIO_PROMPT):
        print(f"Error: {AUDIO_PROMPT} not found. Please upload it.")
        exit(1)
        
    print(f"\n1. Calling prepare_conditionals('{AUDIO_PROMPT}')...")
    latents = tts.prepare_conditionals(AUDIO_PROMPT)
    print(f"   Result type: {type(latents)}")
    if latents:
        print(f"   Result: {latents}")
    
    print("\n2. Checking internal state 'conds'...")
    if hasattr(tts, 'conds'):
        print(f"   tts.conds: {tts.conds}")
    
    print("\n3. Calling generate('Test') WITHOUT audio_prompt_path...")
    try:
        wav = tts.generate("This is a test of caching.")
        print("   ✅ SUCCESS: Generated audio without explicit path.")
        print(f"   Output shape: {wav.shape}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

except Exception as e:
    print(f"Error: {e}")
