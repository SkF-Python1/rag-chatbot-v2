# Quick Setup Guide - RAG Chatbot

## Prerequisites

1. **Python 3.11+** installed
2. **OpenAI API key** (required)
3. **AWS account** (optional, for S3 storage)
4. **Redis** (optional, for caching)

## Quick Start (Local Development)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# At minimum, set OPENAI_API_KEY=your_key_here
```

### 3. Start the Application

```bash
# Option 1: Using the startup script
chmod +x scripts/start.sh
./scripts/start.sh dev

# Option 2: Direct command
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## Docker Deployment

### 1. Using Docker Compose (Recommended)

```bash
# Set environment variables
export OPENAI_API_KEY=your_key_here
export AWS_ACCESS_KEY_ID=your_aws_key
export AWS_SECRET_ACCESS_KEY=your_aws_secret
export S3_BUCKET_NAME=your_bucket_name

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f rag-chatbot
```

### 2. Docker Build Only

```bash
# Build image
docker build -t rag-chatbot .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  rag-chatbot
```

## Configuration

### Required Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Optional Environment Variables

```bash
# AWS (for S3 document storage)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET_NAME=your-documents-bucket

# Performance tuning
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
VECTOR_SEARCH_K=5
MAX_TOKENS_PER_RESPONSE=1000

# Caching
REDIS_URL=redis://localhost:6379
```

## Usage

### 1. Upload Documents

- Drag and drop files in the web interface
- Supported formats: PDF, Word, HTML, Text, Markdown
- Documents are automatically processed and indexed

### 2. Chat with Your Documents

- Ask questions about your uploaded documents
- The system provides context-aware responses
- Source citations show which documents were used

### 3. API Usage

```python
import requests

# Chat endpoint
response = requests.post('http://localhost:8000/api/chat', json={
    'message': 'What is this document about?',
    'use_context': True
})

# Document upload
files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8000/api/documents/upload', files=files)
```

## Performance Features

- **Sub-1.2s Response Time**: Optimized for fast responses
- **FAISS Vector Search**: Efficient similarity search
- **Async Processing**: Non-blocking operations
- **Background Document Processing**: Files processed asynchronously
- **Health Monitoring**: Built-in health checks and metrics

## Troubleshooting

### Common Issues

1. **Dependencies not installing**: Ensure Python 3.11+ and build tools are installed
2. **OpenAI API errors**: Check your API key and rate limits
3. **Slow responses**: Check network connection and consider using Redis for caching
4. **File upload fails**: Check file size limits and supported formats

### Logs

```bash
# View application logs
tail -f logs/rag_chatbot.log

# Docker logs
docker-compose logs -f rag-chatbot
```

### Health Checks

```bash
# Basic health
curl http://localhost:8000/api/health

# Detailed health (includes all services)
curl http://localhost:8000/api/health/detailed
```

## Development

### Project Structure

```
RAG/
├── src/                 # Backend source code
├── frontend/           # Web interface
├── scripts/           # Utility scripts
├── data/             # Local data storage
├── logs/             # Application logs
├── requirements.txt  # Python dependencies
├── docker-compose.yml # Docker configuration
└── README.md        # Project documentation
```

### Adding New Features

1. Backend changes go in `src/`
2. Frontend changes go in `frontend/`
3. Update `requirements.txt` for new dependencies
4. Update API documentation in FastAPI

## Support

- Check the logs for error messages
- Verify all environment variables are set correctly
- Ensure adequate disk space for document processing
- Monitor API rate limits for OpenAI

Happy chatting! 🤖 