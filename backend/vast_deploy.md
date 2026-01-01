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

# 6. Filebase Config
export FILEBASE_KEY="C1A1C1B021991042D1A1"
export FILEBASE_SECRET="C2IpJ7KB6wxBXl6LjWCMX5L5RcHFfYPs9MPcfyAf"
export FILEBASE_BUCKET="hitler-audio" 
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
ssh -p 11064 root@ssh3.vast.ai -L 8080:localhost:8080
```powershell
# 1. Upload Code
scp -P 11064 "C:\Users\OM\.gemini\antigravity\scratch\dictator-ai\dict\backend\gpu_node.py" root@ssh3.vast.ai:/workspace/gpu_node.py

# 2. Upload Reference Audio (CRITICAL)
# Make sure the file exists at this path!
scp -P 11064 "C:\Users\OM\Downloads\reference.wav" root@ssh3.vast.ai:"/workspace/hitler_shouting_clean (1).mp3"
```



## 6. Run Services (On Server)
Back in your `ssh` session:

### Step A: Start LLM
```bash
nohup /workspace/llama.cpp/build/bin/llama-server \
  -m /root/.cache/llama.cpp/dictator.gguf \
  --port 19000 \
  --host 0.0.0.0 \
  -c 8192 \
  -np 4 \
  -cb \
  > /tmp/llama.log 2>&1 &
```

### Step B: Start GPU Node (XTTS)
```bash
cd /workspace
export LLM_URL="http://127.0.0.1:19000/v1/chat/completions"
nohup ./venv/bin/python gpu_node.py > /tmp/gpu_node.log 2>&1 &

# Monitor
tail -f /tmp/gpu_node.log

# killing process
pkill -f gpu_node.py
```

You should see: `✅ XTTS Model Ready.`


## 7. Verification
From your **Local PowerShell**:

```powershell
# Check Health
# IMPORTANT: Use the EXTERNAL PORT mapped to 6000 (e.g. 20111 in your case)
# Check your Vast.ai "Open Ports" section.

Invoke-RestMethod -Uri "http://74.48.140.178:35652/generate_stream" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"messages": [{"role": "user", "content": "hello"}], "tier": "free"}'

```