"""
Document-related data models for the RAG Chatbot.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    
    id: str = Field(..., description="Unique document ID")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type/extension")
    file_size: int = Field(..., description="File size in bytes")
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    s3_key: Optional[str] = Field(default=None, description="S3 object key")
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    chunk_count: int = Field(default=0, description="Number of chunks created")
    
    # Document content metadata
    title: Optional[str] = Field(default=None, description="Document title")
    author: Optional[str] = Field(default=None, description="Document author")
    creation_date: Optional[datetime] = Field(default=None, description="Document creation date")
    language: Optional[str] = Field(default="en", description="Document language")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    
    # Additional metadata
    custom_metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Custom metadata fields"
    )


class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    
    id: str = Field(..., description="Unique chunk ID")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Index of chunk within document")
    start_char: int = Field(..., description="Start character position in original document")
    end_char: int = Field(..., description="End character position in original document")
    
    # Vector embeddings (stored separately in FAISS)
    embedding_id: Optional[str] = Field(default=None, description="Reference to embedding in vector store")
    
    # Chunk metadata
    page_number: Optional[int] = Field(default=None, description="Page number (for PDFs)")
    section: Optional[str] = Field(default=None, description="Section/heading")
    word_count: int = Field(..., description="Word count in chunk")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Document(BaseModel):
    """Complete document representation."""
    
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = Field(default_factory=list)
    raw_content: Optional[str] = Field(default=None, description="Full document text")
    
    def add_chunk(self, content: str, chunk_index: int, start_char: int, end_char: int, **kwargs) -> DocumentChunk:
        """Add a chunk to the document."""
        chunk_id = f"{self.metadata.id}_chunk_{chunk_index}"
        chunk = DocumentChunk(
            id=chunk_id,
            document_id=self.metadata.id,
            content=content,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            word_count=len(content.split()),
            **kwargs
        )
        self.chunks.append(chunk)
        self.metadata.chunk_count = len(self.chunks)
        return chunk
    
    def get_chunk_by_index(self, index: int) -> Optional[DocumentChunk]:
        """Get chunk by index."""
        for chunk in self.chunks:
            if chunk.chunk_index == index:
                return chunk
        return None


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    
    filename: str = Field(..., description="Filename")
    file_type: str = Field(..., description="File MIME type")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    custom_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Custom metadata")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    
    document_id: str = Field(..., description="Unique document ID")
    upload_url: Optional[str] = Field(default=None, description="Pre-signed S3 upload URL")
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")


class DocumentSearchRequest(BaseModel):
    """Request model for document search."""
    
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    document_ids: Optional[List[str]] = Field(default=None, description="Filter by document IDs")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    file_types: Optional[List[str]] = Field(default=None, description="Filter by file types")


class DocumentSearchResult(BaseModel):
    """Search result for a document chunk."""
    
    chunk: DocumentChunk
    score: float = Field(..., description="Similarity score")
    document_metadata: DocumentMetadata 