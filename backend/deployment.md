# Backend Deployment (Vast.ai)

This folder contains the Inference logic (`gpu_node.py`), TTS, and LLM connection.

**Host**: Vast.ai (GPU Instance)

## 1. Create Instance
Use the Vast.ai CLI or UI to find an RTX 3090/4090 instance.
Open ports: `6000` (Node) and `19000` (LLM).

## 2. Setup
SSH into the instance and run:

```bash
# Install Python 3.11 & Venv
apt-get update && apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get install -y python3.11 python3.11-venv

# Setup Workspace
mkdir -p /workspace
cd /workspace
python3.11 -m venv venv
source venv/bin/activate

# Install Deps
pip install flask requests huggingface_hub[cli] chatterbox-tts
# (See vast_deploy.md for full dependency list)
```

## 3. Upload Code
From your local machine:
```powershell
scp -P [PORT] gpu_node.py root@[IP]:/workspace/
scp -P [PORT] static_audio/* root@[IP]:/workspace/static_audio/
```

## 4. Run
```bash
# Start LLM (Llama.cpp)
nohup ./llama-server ... &

# Start Node
export LLM_URL="http://127.0.0.1:19000/v1/chat/completions"
nohup ./venv/bin/python gpu_node.py > gpu.log 2>&1 &
```
