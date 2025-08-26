#!/bin/bash

# RAG Chatbot Startup Script

set -e

echo "🤖 Starting RAG Chatbot Application..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "⚠️  No .env file found. Copying from .env.example..."
        cp .env.example .env
        echo "📝 Please edit .env file with your configuration and restart."
        exit 1
    else
        echo "❌ No .env file found and no .env.example available."
        exit 1
    fi
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs data/faiss_index

# Check if Python dependencies are installed
echo "🔍 Checking Python dependencies..."
if ! python -c "import fastapi, openai, langchain, faiss" 2>/dev/null; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Check system health
echo "🏥 Checking system health..."
if command -v redis-cli &> /dev/null; then
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is running"
    else
        echo "⚠️  Redis is not running. Some caching features may not work."
    fi
else
    echo "⚠️  Redis not found. Some caching features may not work."
fi

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    if grep -q "OPENAI_API_KEY=" .env; then
        echo "✅ OpenAI API key found in .env"
    else
        echo "❌ OpenAI API key not configured. Please set OPENAI_API_KEY in .env"
        exit 1
    fi
fi

# Start the application
echo "🚀 Starting FastAPI application..."

if [ "$1" = "dev" ]; then
    echo "🔧 Running in development mode..."
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
elif [ "$1" = "prod" ]; then
    echo "🏭 Running in production mode..."
    uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
else
    echo "💻 Running in standard mode..."
    uvicorn src.main:app --host 0.0.0.0 --port 8000
fi 