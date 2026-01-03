#!/bin/bash

# 1. Set Environment (If not already set in ~/.bashrc or Vast env)
# You should set these in .env file or Vast Env Vars for security
# export MONGO_URI="..."
# export FILEBASE_KEY="..."
# export FILEBASE_SECRET="..."
export INTERNAL_SECRET="DICTATOR_INTERNAL_V1_SECURE"
# CRITICAL: Point to the LLM on Port 19000
export LLM_URL="http://127.0.0.1:19000/v1/chat/completions"

# Hugging Face Audio Storage (Private)
export HF_TOKEN="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"
export HF_DATASET="krishnasuratwala/dictator-ai-audio"

echo "🚀 Starting Dictator AI Fortress..."


# 1.5 Activate Virtual Environment (CRITICAL)
if [ -d "venv" ]; then
    echo "🐍 Activating venv..."
    source venv/bin/activate
elif [ -d "/workspace/venv" ]; then
    echo "🐍 Activating /workspace/venv..."
    source /workspace/venv/bin/activate
else
    echo "⚠️  No venv found! Running with system python (might fail)..."
fi

# 1.6 Start LLM Server (Background, Port 19000)
echo "🧠 Starting LLM Server (Port 19000)..."
# Use stdbuf -oL to force line buffering for logs
if command -v stdbuf &> /dev/null; then
    LLM_CMD="stdbuf -oL /workspace/llama.cpp/build/bin/llama-server"
else
    LLM_CMD="/workspace/llama.cpp/build/bin/llama-server"
fi

nohup $LLM_CMD \
  -m /root/.cache/llama.cpp/dictator.gguf \
  --port 19000 \
  --host 0.0.0.0 \
  -c 8192 \
  -np 4 \
  -cb \
  > llama_server.log 2>&1 &
LLM_PID=$!
echo "✅ LLM Server PID: $LLM_PID"

# Wait for LLM to warm up (optional but good)
echo "⏳ Waiting 5s for LLM to initialize..."
sleep 5

# 2. Start Backend (Private, Background, Port 5000)
# > /dev/null 2>&1 hides output, remove for debugging
echo "Starting GPU Node (Private Port 5000)..."
# Use -u for unbuffered python output
nohup python3 -u gpu_node.py > gpu_node.log 2>&1 &
GPU_PID=$!
echo "✅ GPU Node PID: $GPU_PID"

# Wait a bit for model load
# 2.5 Install Cloudflare Tunnel (The Network Fix)
if [ ! -f "cloudflared-linux-amd64.deb" ]; then
    echo "☁️ Installing Cloudflare Tunnel..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    dpkg -i cloudflared-linux-amd64.deb
fi

# 3. Start Middleware (Public, Foreground, HTTP)
# Gunicorn binds to 6000 (HTTP)
# We rely on Cloudflare to provide HTTPS
echo "Starting Middleware (HTTP Port 6000)..."
# Start Cloudflare Tunnel in Background
nohup cloudflared tunnel --url http://127.0.0.1:6000 > tunnel.log 2>&1 &
echo "🚇 Cloudflare Tunnel Started..."

# Helper: Find and Print URL (Background)
(
  sleep 5
  echo "🔍 Scanning for Public URL..."
  for i in {1..20}; do
    URL=$(grep -o 'https://[^"]*\.trycloudflare\.com' tunnel.log | head -n 1)
    if [ ! -z "$URL" ]; then
      echo ""
      echo "========================================================"
      echo "🦄 YOUR PUBLIC URL: $URL"
      echo "========================================================"
      echo ""
      break
    fi
    sleep 2
  done
) &

# 4 Workers, 4 Threads
gunicorn -w 4 --threads 4 -b 0.0.0.0:6000 server:app

# Cleanup on exit
# Handle both PIDs
kill $GPU_PID
kill $LLM_PID
