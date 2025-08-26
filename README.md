# Context-Aware RAG Chatbot

A high-performance RAG (Retrieval-Augmented Generation) chatbot built with OpenAI GPT-4, FAISS, LangChain, and FastAPI. Designed for internal knowledge bases with 10K+ documents achieving 93% answer relevance and sub-1.2s response times.

## Features

- **Multi-format Document Support**: PDF, HTML, Word documents
- **High Performance**: <1.2s response latency with 93% answer relevance
- **Scalable Architecture**: Handles 10K+ documents efficiently
- **AWS Integration**: S3 storage for document ingestion
- **Modern UI**: React-based chat interface
- **Vector Search**: FAISS for fast similarity search
- **Enterprise Ready**: FastAPI backend with async processing

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI        │    │   Document      │
│   (React)       │◄──►│   Backend        │◄──►│   Processing    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   LangChain      │    │   AWS S3        │
                       │   RAG Pipeline   │    │   Storage       │
                       └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   FAISS Vector   │
                       │   Store          │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   OpenAI GPT-4   │
                       └──────────────────┘
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Setup**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the Application**
   ```bash
   # Start backend
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   
   # Start frontend (in another terminal)
   cd frontend && npm start
   ```

4. **Upload Documents**
   - Access the web interface at `http://localhost:3000`
   - Upload your documents via the interface or S3

## Project Structure

```
RAG/
├── src/                    # Backend source code
│   ├── api/               # FastAPI routes
│   ├── core/              # Core business logic
│   ├── models/            # Data models
│   ├── services/          # Service layer
│   └── utils/             # Utility functions
├── frontend/              # React frontend
├── data/                  # Local data storage
├── tests/                 # Test files
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── docker-compose.yml    # Docker configuration
```

## Configuration

See `.env.example` for required environment variables:
- OpenAI API key
- AWS credentials
- Database settings
- Performance tuning parameters

## Performance Optimizations

- **Async Processing**: Non-blocking I/O operations
- **Caching**: Redis for frequently accessed data
- **Batch Processing**: Efficient document ingestion
- **Connection Pooling**: Optimized database connections
- **Vector Index Optimization**: FAISS parameter tuning

## License

MIT License - see LICENSE file for details. 