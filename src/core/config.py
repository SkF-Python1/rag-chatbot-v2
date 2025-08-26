"""
Configuration settings for the RAG Chatbot application.
"""

from typing import List, Optional
from pydantic import BaseSettings, Field
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="OpenAI model to use")
    openai_embedding_model: str = Field(
        default="text-embedding-ada-002", 
        description="OpenAI embedding model"
    )
    
    # AWS Configuration
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret key")
    aws_region: str = Field(default="us-east-1", description="AWS region")
    s3_bucket_name: Optional[str] = Field(default=None, description="S3 bucket for documents")
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./rag_chatbot.db", 
        description="Database URL"
    )
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    redis_cache_ttl: int = Field(default=3600, description="Redis cache TTL in seconds")
    
    # FastAPI Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        description="CORS allowed origins"
    )
    
    # Vector Store Configuration
    faiss_index_path: str = Field(
        default="./data/faiss_index", 
        description="Path to FAISS index"
    )
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Text chunk overlap")
    max_documents: int = Field(default=10000, description="Maximum documents to index")
    
    # Performance Settings
    max_concurrent_requests: int = Field(
        default=10, 
        description="Maximum concurrent requests"
    )
    response_timeout: int = Field(default=10, description="Response timeout in seconds")
    vector_search_k: int = Field(default=5, description="Number of vectors to retrieve")
    max_tokens_per_response: int = Field(
        default=1000, 
        description="Maximum tokens per response"
    )
    
    # Document Processing
    supported_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".html", ".txt", ".md"],
        description="Supported file extensions"
    )
    max_file_size_mb: int = Field(default=50, description="Maximum file size in MB")
    document_processing_batch_size: int = Field(
        default=10, 
        description="Batch size for document processing"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_file: str = Field(default="./logs/rag_chatbot.log", description="Log file path")
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30, 
        description="Access token expiration in minutes"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)


# Global settings instance
settings = Settings() 