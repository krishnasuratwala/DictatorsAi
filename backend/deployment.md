# Vast.ai Deployment (XTTS Edition)

> [!NOTE]
> This guide details how to deploy the **Inference Node** (`gpu_node.py`) with **Coqui XTTS** and the **LLM Server** (`llama-server`) on a Vast.ai GPU instance.

## 1. Create Instance
Run this from your **Local PowerShell**:

```powershell
# Search for a good consumer GPU (RTX 3090/4090/A6000)
# Ensure > 40GB Disk Space for XTTS models
vastai search offers ' gpu_ram>=24 verified=true rentable=true disk_space>=40'

# Create Instance (Open Ports 22, 6000, 19000)
vastai create instance 29204625       `
  --image vastai/base-image:@vastai-automatic-tag `
  --env '-p 22:22 -p 6000:6000 -p 19000:19000' `
  --disk 40 `
  --ssh `
  --direct
```

## 2. Connect
```powershell
# Replace [PORT] and [IP] with your instance details
ssh -p [PORT] root@[IP] -L 6000:localhost:6000
```

## 3. Setup Environment (On Server)
Run these commands inside the `ssh` session:

```bash
# 1. System Dependencies & Python 3.11 (Required for Coqui TTS < 3.12)
apt-get update && apt-get install -y git cmake build-essential wget curl software-properties-common ffmpeg libavutil-dev
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev libcurl4-openssl-dev

# 2. Setup Workspace
mkdir -p /workspace
cd /workspace
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 3. Install PyTorch (Nightly required for RTX 50-Series / sm_120)
pip install --pre --upgrade --force-reinstall torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/nightly/cu128
# FIX: TorchCodec requires NVIDIA NPP libraries for video/audio decoding
pip install nvidia-npp-cu12

# FIX: Add NVIDIA libs to path so TorchCodec can find them
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/venv/lib/python3.11/site-packages/nvidia/npp/lib:/workspace/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/workspace/venv/lib/python3.11/site-packages/nvidia/cufft/lib

# 4. Install Application Dependencies
pip install flask requests boto3
pip install "numpy<2" "pillow<11" 
pip install scipy soundfile tqdm huggingface_hub[cli] vector-quantize-pytorch 
pip install sacremoses sentencepiece 

# 5. Install Coqui TTS & Fix Transformers Version
# IMPORTANT: Accept the license if prompted or ensure --no-input
pip install TTS
# FIX: Downgrade transformers to fix 'BeamSearchScorer' import error in older TTS versions
pip install "transformers==4.40.1" "accelerate>=0.26.0"

# # 6. Filebase Config
# export FILEBASE_KEY="C1A1C1B021991042D1A1"
# export FILEBASE_SECRET="C2IpJ7KB6wxBXl6LjWCMX5L5RcHFfYPs9MPcfyAf"
# export FILEBASE_BUCKET="hitler-audio" 
```

## 4. Download Models

### A. Download XTTS Fine-Tuned Model
```bash
# Login to Hugging Face (Optional if repo is public, but good practice)
huggingface-cli login --token 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

# Download Model to /workspace/xtts_model
# Repo: krishnasuratwala/XTTS_HITLER_AUDIO
hf download krishnasuratwala/XTTS_HITLER_AUDIO --local-dir /workspace/xtts_model
```

### B. Download & Build Llama.cpp (LLM)
```bash
cd /workspace
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir -p build
cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . --config Release -j$(nproc)
```

### C. Download LLM Weights
```bash
mkdir -p /root/.cache/llama.cpp
wget --continue --progress=bar:force:noscroll \
  -O /root/.cache/llama.cpp/dictator.gguf \
  "https://huggingface.co/krishnasuratwala/Mistral-7B-Instruct-v0.2-FineTuned-GGUF/resolve/main/Final_Hitler_model.3.Q5_K_M.gguf"

# Quantize if needed (Optional)
# ./llama-quantize /root/.cache/llama.cpp/dictator.gguf /root/.cache/llama.cpp/dictator-q4.gguf Q4_K_M
```

## 5. Upload Code & Audio (From Local)
Open a **new local terminal**.
Replace `[PORT]` and `[IP]` with your Vast details (e.g. `ssh -p 58413 root@199.126.134.31`).

```powershell
# 1. Upload Core Code
scp -P [PORT] "C:\Users\OM\.gemini\antigravity\scratch\dictator-ai\dict\backend\gpu_node.py" root@[IP]:/workspace/gpu_node.py
scp -P [PORT] "C:\Users\OM\.gemini\antigravity\scratch\dictator-ai\dict\middleware\server.py" root@[IP]:/workspace/server.py
scp -P [PORT] "C:\Users\OM\.gemini\antigravity\scratch\dictator-ai\dict\backend\requirements.txt" root@[IP]:/workspace/requirements.txt
scp -P [PORT] "C:\Users\OM\.gemini\antigravity\scratch\dictator-ai\dict\backend\start_all.sh" root@[IP]:/workspace/start_all.sh
scp -P [PORT] "C:\Users\OM\.gemini\antigravity\scratch\dictator-ai\dict\middleware\.env" root@[IP]: /workspace/.env
scp -P [PORT] "C:\Users\OM\.gemini\antigravity\scratch\dictator-ai\dict\middleware\btcpay_utils.py" root@[IP]:/workspace/btcpay_utils.py

# 2. Upload Assets (Video & Audio)
scp -P [PORT] "C:\Users\OM\.gemini\antigravity\scratch\dictator-ai\dict\frontend\src\assets\Branding_Video_Generation_Prompt.mp4" root@[IP]:/workspace/Branding_Video_Generation_Prompt.mp4
scp -P [PORT] "C:\Users\OM\Downloads\reference.wav" root@[IP]:"/workspace/hitler_shouting_clean (1).mp3"
```

## 6. Run Services (On Server)
Back in your `ssh` session, you have two options:

### Option A: The "One-Click" (Production)
Runs everything (LLM, Backend, Middleware, Tunnel) in the background.

```bash
cd /workspace
pip install -r requirements.txt
chmod +x start_all.sh
./start_all.sh
ctrl+z
bg
tail -f gpu_node.log
```
*Wait 10-20s. It will print your `trycloudflare.com` URL to the screen.*

# kill process
```bash
pkill -f gunicorn
pkill -f gpu_node.py
pkill -f cloudflared
``` 






### Option B: The "Debug" (Manual)
Best for seeing errors live. Open **Two Terminal Windows**.

**Terminal 1: The Muscle (Backend)**
```bash
cd /workspace
source venv/bin/activate
export INTERNAL_SECRET="DICTATOR_INTERNAL_V1_SECURE"
export LLM_URL="http://127.0.0.1:19000/v1/chat/completions"

# 1. Start LLM (Background) if not running
nohup /workspace/llama.cpp/build/bin/llama-server \
  -m /root/.cache/llama.cpp/dictator.gguf \
  --port 19000 --host 127.0.0.1 -c 8192 -np 4 -cb \
  > /tmp/llama.log 2>&1 &

# 2. Run Backend (Foreground)
python3 gpu_node.py
```

**Terminal 2: The Brain (Middleware)**
```bash
cd /workspace
source venv/bin/activate
export INTERNAL_SECRET="DICTATOR_INTERNAL_V1_SECURE"

# 1. Start Tunnel
nohup cloudflared tunnel --url http://127.0.0.1:6000 > tunnel.log 2>&1 &
grep "trycloudflare.com" tunnel.log

# 2. Start Server
gunicorn -w 1 --threads 4 -b 0.0.0.0:6000 server:app --access-logfile -
```