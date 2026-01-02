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

from huggingface_hub import HfApi

# --- HUGGING FACE CONFIG ---
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET = os.getenv("HF_DATASET")

def upload_to_hf(local_path, filename):
    if not HF_TOKEN or not HF_DATASET: 
        print("⚠️ HF Token/Dataset missing.")
        return None
    try:
        api = HfApi()
        print(f"⬆️  Uploading {filename} to Hugging Face ({HF_DATASET})...")
        
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=HF_DATASET,
            repo_type="dataset",
            token=HF_TOKEN
        )
        
        # Return a Magic Prefix for Middleware to rewrite
        # Middleware will replace 'HF_PROXY:' with its own public URL + /api/audio/
        magic_url = f"HF_PROXY:{filename}"
        print(f"✅ Uploaded. Returning Proxy Key: {magic_url}")
        return magic_url
        
    except Exception as e:
        print(f"❌ HF Upload Failed: {e}")
        return None 

def generate_voice(text, filename, prompt_path):
    global tts_model, xtts_config
    import numpy as np
    
    if not tts_model: 
        print("❌ TTS Model not loaded.")
        return None
    if not os.path.exists(prompt_path):
        print(f"❌ Audio prompt not found: {prompt_path}")
        return None
        
    try:
        print(f"🎙️ Generating Audio (DE): '{text[:50]}...' using {prompt_path}")
        
        # 1. Split Text into readable chunks (XTTS Limit ~250 chars)
        # 1. Split Text into small, digestible chunks (XTTS Limit ~100-120 chars for stability)
        def split_text(text, max_len=120):
            # Sanitize: Replace complex dashes with simple commas
            text = text.replace("–", ",").replace("—", ",").replace("-", ",")
            
            # Aggressive Splitting Pattern: Split on . ! ? ; : and even , if needed
            # We split by these delimiters but keep them attached to the previous word if possible
            import re
            
            # Step 1: Split by major punctuation first
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            final_chunks = []
            
            for sentence in sentences:
                if len(sentence) < max_len:
                    final_chunks.append(sentence.strip())
                else:
                    # Step 2: If sentence is too long, split by clauses (comma, semicolon, colon)
                    sub_parts = re.split(r'(?<=[,;:])\s+', sentence)
                    current_chunk = ""
                    
                    for part in sub_parts:
                        if len(current_chunk) + len(part) < max_len:
                            current_chunk += part + " "
                        else:
                            if current_chunk.strip():
                                final_chunks.append(current_chunk.strip())
                            current_chunk = part + " "
                    
                    if current_chunk.strip():
                        final_chunks.append(current_chunk.strip())
            
            # Filter empty strings
            return [c for c in final_chunks if c.strip()]

        chunks = split_text(text)
        all_wav_parts = []
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip(): continue
            print(f"   🔹 Processing Chunk {i+1}/{len(chunks)} ({len(chunk)} chars): '{chunk}'")
            
            try:
                outputs = tts_model.synthesize(
                    chunk,
                    xtts_config,
                    speaker_wav=prompt_path,
                    gpt_cond_len=3,
                    language="de",
                    temperature=0.75,
                    length_penalty=1.0,
                    repetition_penalty=2.0,
                    top_k=50,
                    top_p=0.85
                )
                wav_chunk = outputs['wav']
                print(f"      ✅ Generated {len(wav_chunk)} samples.")
                all_wav_parts.append(wav_chunk)
                
                # Add small silence (0.2s) between sentences
                silence_samples = int(xtts_config.audio.sample_rate * 0.2)
                all_wav_parts.append(np.zeros(silence_samples, dtype=np.float32))
            except Exception as ce:
                print(f"      ⚠️ Chunk Generation Failed: {ce}")

        if not all_wav_parts: return None
        
        # 2. Concatenate all audio parts
        final_wav = np.concatenate(all_wav_parts)
        
        # 3. Save output
        path = os.path.join("static_audio", filename)
        wav_tensor = torch.from_numpy(final_wav).unsqueeze(0)
        torchaudio.save(path, wav_tensor, xtts_config.audio.sample_rate)
        
        return upload_to_hf(path, filename) or f"/static_audio/{filename}"
    except Exception as e:
        print(f"❌ XTTS Generation Error: {e}")
        return None

from functools import wraps

# --- INTERNAL SECURITY ---
INTERNAL_SECRET = os.getenv("INTERNAL_SECRET", "DICTATOR_INTERNAL_V1_SECURE")

def require_internal_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('X-Internal-Secret')
        if not token or token != INTERNAL_SECRET:
            print(f"🛑 Unauthorized Access Attempt from {request.remote_addr}")
            return "Unauthorized", 401
        return f(*args, **kwargs)
    return decorated

@app.route('/generate_stream', methods=['POST'])
@require_internal_token
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
                    "n_predict": 516, 
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
                    # Split input text if too long for translator (limit ~512 tokens)
                    import re
                    # Split by sentence endings to keep context
                    input_sentences = re.split(r'(?<=[.!?])\s+', full_response_text)
                    translated_parts = []
                    
                    # Batch translate if possible, or loop
                    # HuggingFace pipeline handles batching but safest to loop for error isolation
                    for sent in input_sentences:
                        if not sent.strip(): continue
                        try:
                            res = translator(sent)
                            if res: translated_parts.append(res[0]['translation_text'])
                        except Exception as te:
                            print(f"⚠️ Translation partial fail: {te}")
                            translated_parts.append(sent) # Fallback to English
                            
                    final_text = " ".join(translated_parts)
                    print(f"✅ Translated: {final_text}... (Length: {len(final_text)})")
                
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
    load_xtts_model()
    print("🚀 GPU Node (Streaming) starting on port 5000...")
    # SECURE: Bind to Localhost Only.
    # Only Middleware on the same machine can access this.
    app.run(host='127.0.0.1', port=5000, threaded=True)
