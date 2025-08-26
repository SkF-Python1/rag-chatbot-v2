"""
Main FastAPI application for the RAG Chatbot.
"""

import os
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Local imports
from .core.config import settings
from .services.document_processor import DocumentProcessor
from .services.vector_store import VectorStore
from .services.rag_service import RAGService
from .services.s3_service import S3Service
from .models.chat import ChatRequest, ChatResponse
from .models.document import (
    DocumentUploadRequest, DocumentUploadResponse, 
    DocumentSearchRequest, DocumentSearchResult,
    DocumentMetadata
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global service instances
vector_store: VectorStore = None
rag_service: RAGService = None
document_processor: DocumentProcessor = None
s3_service: S3Service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting RAG Chatbot application...")
    
    global vector_store, rag_service, document_processor, s3_service
    
    try:
        # Initialize services
        vector_store = VectorStore()
        document_processor = DocumentProcessor()
        s3_service = S3Service()
        rag_service = RAGService(vector_store)
        
        # Create S3 bucket if needed
        if settings.s3_bucket_name:
            await s3_service.create_bucket_if_not_exists()
        
        # Optimize vector store
        await vector_store.optimize_index()
        
        logger.info("RAG Chatbot application started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Chatbot application...")


# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A high-performance RAG chatbot for internal knowledge bases",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files (for frontend)
if os.path.exists("frontend/build"):
    app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")


# Dependency injection
def get_vector_store() -> VectorStore:
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    return vector_store


def get_rag_service() -> RAGService:
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    return rag_service


def get_document_processor() -> DocumentProcessor:
    if document_processor is None:
        raise HTTPException(status_code=500, detail="Document processor not initialized")
    return document_processor


def get_s3_service() -> S3Service:
    if s3_service is None:
        raise HTTPException(status_code=500, detail="S3 service not initialized")
    return s3_service


# Chat endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Process a chat message and return a response using RAG.
    Optimized for sub-1.2s response time.
    """
    try:
        start_time = time.time()
        
        # Validate request
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Process chat request
        response = await rag_service.chat(request)
        
        # Log performance
        total_time = time.time() - start_time
        logger.info(f"Chat request processed in {total_time:.3f}s")
        
        if total_time > 1.2:
            logger.warning(f"Response time {total_time:.3f}s exceeded target of 1.2s")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get conversation history by ID."""
    try:
        conversation = await rag_service.get_conversation_history(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/api/conversations/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Clear conversation history."""
    try:
        success = await rag_service.clear_conversation(conversation_id)
        return {"success": success}
        
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Document management endpoints
@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    s3_service: S3Service = Depends(get_s3_service),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Upload and process a document."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}"
            )
        
        # Read file content
        content = await file.read()
        if len(content) > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Generate document ID
        import uuid
        document_id = str(uuid.uuid4())
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        
        # Upload to S3 if configured
        s3_key = None
        if settings.s3_bucket_name:
            s3_key = await s3_service.upload_document(document_id, file.filename, content)
        
        # Schedule background processing
        background_tasks.add_task(
            process_document_background,
            document_id, 
            file.filename, 
            content, 
            tag_list,
            s3_key
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            status="uploaded",
            message="Document uploaded and processing started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def process_document_background(
    document_id: str,
    filename: str,
    content: bytes,
    tags: List[str],
    s3_key: Optional[str]
):
    """Background task to process uploaded document."""
    try:
        # Save file temporarily
        temp_path = f"./data/temp_{document_id}_{filename}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        # Process document
        document = await document_processor.process_document(
            temp_path, 
            filename, 
            tags=tags,
            custom_metadata={"s3_key": s3_key} if s3_key else None
        )
        
        # Add to vector store
        success = await vector_store.add_document_chunks(document.chunks)
        
        if success:
            logger.info(f"Successfully processed document {filename}")
            
            # Copy to processed folder in S3
            if s3_key and s3_service:
                await s3_service.copy_to_processed(s3_key, document_id)
        else:
            logger.error(f"Failed to add document {filename} to vector store")
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        logger.error(f"Error processing document {filename}: {str(e)}")


@app.get("/api/documents/search")
async def search_documents(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum results"),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Search documents using semantic similarity."""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        results = await vector_store.search_similar_chunks(query, k=limit)
        
        return {
            "query": query,
            "results": [
                {
                    "chunk_id": result.chunk.id,
                    "document_id": result.chunk.document_id,
                    "content": result.chunk.content,
                    "score": result.score,
                    "page_number": result.chunk.page_number,
                    "section": result.chunk.section
                }
                for result in results
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/documents/stats")
async def get_document_stats(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get document statistics."""
    try:
        stats = vector_store.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Health check endpoints
@app.get("/api/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }


@app.get("/api/health/detailed")
async def detailed_health_check(
    rag_service: RAGService = Depends(get_rag_service),
    s3_service: S3Service = Depends(get_s3_service)
):
    """Detailed health check of all services."""
    try:
        # Run health checks in parallel
        rag_health, s3_health = await asyncio.gather(
            rag_service.health_check(),
            s3_service.health_check(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(rag_health, Exception):
            rag_health = {"status": "unhealthy", "error": str(rag_health)}
        
        if isinstance(s3_health, Exception):
            s3_health = {"status": "unhealthy", "error": str(s3_health)}
        
        overall_status = "healthy" if all([
            rag_health.get("status") == "healthy",
            s3_health.get("status") == "healthy"
        ]) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "services": {
                "rag_service": rag_health,
                "s3_service": s3_health
            }
        }
        
    except Exception as e:
        logger.error(f"Error in detailed health check: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }


# Performance monitoring
@app.get("/api/metrics")
async def get_metrics(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get performance metrics."""
    try:
        stats = vector_store.get_stats()
        
        # Add additional metrics
        metrics = {
            "vector_store": stats,
            "system": {
                "timestamp": time.time(),
                "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Root endpoint for frontend
@app.get("/")
async def root():
    """Serve frontend or API info."""
    if os.path.exists("frontend/build/index.html"):
        from fastapi.responses import FileResponse
        return FileResponse("frontend/build/index.html")
    else:
        return {
            "message": "RAG Chatbot API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/health"
        }


# Error handlers
@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Not found"}
    )


# Add startup time tracking
@app.on_event("startup")
async def startup_event():
    app.state.start_time = time.time()


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        workers=1 if settings.debug else 4
    ) 