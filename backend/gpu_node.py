from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
import flask 
import os
import json
import boto3
import torch
import torchaudio
import time
import requests
from botocore.client import Config
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from transformers import pipeline

# --- CONFIGURATION ---
app = Flask(__name__, static_folder='static_audio')
os.makedirs('static_audio', exist_ok=True)

LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:5000/v1/chat/completions") 
CUSTOM_MODEL = "krishnasuratwala/Dictatorai_one_to_one_Q4_K_M:Q4_K_M"
# Update referencing the new required audio file
AUDIO_PROMPT_REF = "hitler_shouting_clean (1).mp3"

# --- FILEBASE CONFIG ---
FILEBASE_KEY = os.getenv("FILEBASE_KEY", "C1A1C1B021991042D1A1")
FILEBASE_SECRET = os.getenv("FILEBASE_SECRET", "C2IpJ7KB6wxBXl6LjWCMX5L5RcHFfYPs9MPcfyAf")
FILEBASE_BUCKET = os.getenv("FILEBASE_BUCKET", "hitler-audio") 

s3_client = None
if FILEBASE_KEY and FILEBASE_SECRET:
    try:
        s3_client = boto3.client('s3',
            endpoint_url='https://s3.filebase.com',
            aws_access_key_id=FILEBASE_KEY,
            aws_secret_access_key=FILEBASE_SECRET,
            config=Config(signature_version='s3v4')
        )
        print("✅ Filebase S3 Client Configured.")
    except Exception as e:
        print(f"⚠️ Filebase Config Failed: {e}")

# --- TRANSLATION LOADER ---
translator = None
try:
    print("⏳ Loading Translator (en->de)...")
    translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de", device=0 if torch.cuda.is_available() else -1)
    print("✅ Translator Ready.")
except Exception as e:
    print(f"⚠️ Translator Load Failed: {e}")

# --- XTTS LOADER ---
tts_model = None
xtts_config = None

def load_xtts_model():
    global tts_model, xtts_config
    model_path = "/workspace/xtts_model"
    
    if not os.path.exists(model_path):
        print(f"⚠️ XTTS Model path not found: {model_path}")
        return

    try:
        print(f"⏳ Loading XTTS Model from {model_path}...")
        
        # Load Config
        config_path = os.path.join(model_path, "config.json")
        xtts_config = XttsConfig()
        xtts_config.load_json(config_path)
        
        # Init Model
        tts_model = Xtts.init_from_config(xtts_config)
        
        # Resolve Checkpoint File
        # User repo has "unoptimize_model.pth", default is "model.pth"
        checkpoint_filename = "unoptimize_model.pth"
        if not os.path.exists(os.path.join(model_path, checkpoint_filename)):
            checkpoint_filename = "model.pth"
            
        checkpoint_path = os.path.join(model_path, checkpoint_filename)
        vocab_path = os.path.join(model_path, "vocab.json")
        speaker_path = os.path.join(model_path, "speakers_xtts.pth")
        
        print(f"   - Checkpoint: {checkpoint_filename}")
        
        # Load Checkpoint with specific paths
        tts_model.load_checkpoint(
            xtts_config, 
            checkpoint_path=checkpoint_path, 
            vocab_path=vocab_path,
            speaker_file_path=speaker_path, # Explicitly load speaker file usually found in FT repos
            eval=True
        )
        
        # Move to GPU
        if torch.cuda.is_available():
            tts_model.cuda()
            print("⚙️  XTTS Loaded on CUDA")
        else:
            print("⚠️  CUDA not available, loading on CPU (Slow!)")

        print("✅ XTTS Model Ready.")
        
    except Exception as e:
        print(f"❌ Failed to load XTTS Model: {e}")

# Initial Load
load_xtts_model()

def upload_to_filebase(local_path, filename):
    if not s3_client: return None
    try:
        # 1. Upload
        print(f"⬆️  Uploading {filename} to Filebase...")
        s3_client.upload_file(
            local_path, 
            FILEBASE_BUCKET, 
            filename, 
            ExtraArgs={'ContentType': 'audio/wav'}
        )
        
        # 2. Generate SIGNED URL
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': FILEBASE_BUCKET, 'Key': filename},
            ExpiresIn=3600  # 1 Hour Expiry
        )
        
        print(f"✅ Generated Signed URL: {url}")
        return url
        
    except Exception as e:
        print(f"❌ Upload Failed: {e}")
        return None 

def generate_voice(text, filename, prompt_path):
    global tts_model, xtts_config
    if not tts_model: 
        print("❌ TTS Model not loaded.")
        return None
    if not os.path.exists(prompt_path):
        print(f"❌ Audio prompt not found: {prompt_path}")
        return None
        
    try:
        print(f"🎙️ Generating Audio (DE): '{text}' using {prompt_path}")
        
        # Run Inference
        outputs = tts_model.synthesize(
            text,
            xtts_config,
            speaker_wav=prompt_path,
            gpt_cond_len=3,
            language="de"
        )
        
        # Save output
        path = os.path.join("static_audio", filename)
        # outputs['wav'] is numpy array, sample_rate is in config
        # torchaudio.save expects tensor: (channels, time)
        wav_tensor = torch.from_numpy(outputs['wav']).unsqueeze(0)
        torchaudio.save(path, wav_tensor, xtts_config.audio.sample_rate)
        
        return upload_to_filebase(path, filename) or f"/static_audio/{filename}"
    except Exception as e:
        print(f"❌ XTTS Generation Error: {e}")
        return None

@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    data = request.json
    messages = data.get('messages', [])
    tier = data.get('tier', 'free')
    
    # Always use the clean shouting reference as requested
    voice_path = AUDIO_PROMPT_REF
    
    def generate():
        full_response_text = ""
        
        # --- STEP 1: STREAM TEXT FROM LLM ---
        try:
            prompt = data.get('prompt')
            print(f"--- INCOMING PROMPT ---\n{prompt if prompt else json.dumps(messages, indent=2)}\n-----------------------")
            
            if prompt:
                # RAW COMPLETION MODE
                target_url = LLM_URL.replace("/v1/chat/completions", "/completion")
                payload = {
                    "prompt": prompt,
                    "model": CUSTOM_MODEL,
                    "stream": True,
                    "stop": ["<|im_end|>"], 
                    "n_predict": 256, 
                    "temperature": 0.75,
                    "top_p": 0.9,
                    "repeat_penalty": 1.2,
                    "cache_prompt": True 
                }
            else:
                # STANDARD CHAT MODE
                target_url = LLM_URL
                payload = {
                    "model": CUSTOM_MODEL,
                    "messages": messages,
                    "stream": True,
                    "cache_prompt": True 
                }
            
            with requests.post(target_url, json=payload, stream=True, timeout=120) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            json_str = decoded_line[6:] # Strip 'data: '
                            if json_str.strip() == '[DONE]': break
                            try:
                                chunk = json.loads(json_str)
                                content = ""
                                
                                # Handle /completion format
                                if 'content' in chunk:
                                    content = chunk['content']
                                # Handle /v1/chat/completions format
                                elif 'choices' in chunk:
                                    choice = chunk['choices'][0]
                                    if 'delta' in choice and 'content' in choice['delta']:
                                        content = choice['delta']['content']
                                    elif 'text' in choice: # /v1/completions sometimes
                                        content = choice['text']
                                
                                if content:
                                    full_response_text += content
                                    # Yield Text Chunk
                                    yield json.dumps({"type": "text", "content": content}) + "\n"
                            except:
                                pass
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            yield json.dumps({"type": "text", "content": " [CONNECTION INTERRUPTED]"}) + "\n"
            full_response_text = "Connection Interrupted."

        # --- STEP 2: TRANSLATE & GENERATE AUDIO (After Text is Complete) ---
        if tier in ['infantry', 'commander'] and full_response_text:
            try:
                # Translate to German
                final_text = full_response_text
                if translator:
                    print(f"🔄 Translating: {full_response_text[:50]}...")
                    translation = translator(full_response_text)
                    if translation and len(translation) > 0:
                        final_text = translation[0]['translation_text']
                        print(f"✅ Translated: {final_text[:50]}...")
                
                filename = f"speech_{int(time.time())}_{os.urandom(4).hex()}.wav"
                # Generate with German text
                audio_url = generate_voice(final_text, filename, voice_path)
                
                if audio_url:
                    yield json.dumps({"type": "audio", "url": audio_url}) + "\n"
            except Exception as e:
                print(f"❌ Translation/Audio Error: {e}")

    return Response(stream_with_context(generate()), content_type='application/x-ndjson')

# Serve static for fallback
@app.route('/static_audio/<path:filename>')
def serve_audio(filename):
    return flask.send_from_directory('static_audio', filename)

if __name__ == '__main__':
    print("🚀 GPU Node (Streaming) starting on port 6000...")
    app.run(host='0.0.0.0', port=6000, threaded=True)
