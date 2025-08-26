"""
Chat-related data models for the RAG Chatbot.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Represents a single chat message."""
    
    id: Optional[str] = Field(default=None, description="Unique message ID")
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")
    use_context: bool = Field(default=True, description="Whether to use RAG context")
    max_tokens: Optional[int] = Field(default=None, description="Maximum response tokens")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    
    response: str = Field(..., description="Assistant response")
    conversation_id: str = Field(..., description="Conversation ID")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Source documents used for context"
    )
    response_time: float = Field(..., description="Response time in seconds")
    token_count: int = Field(..., description="Number of tokens in response")


class ChatHistory(BaseModel):
    """Represents a conversation history."""
    
    conversation_id: str = Field(..., description="Unique conversation ID")
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation."""
        message = ChatMessage(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message
    
    def get_context_messages(self, max_messages: int = 10) -> List[ChatMessage]:
        """Get recent messages for context."""
        return self.messages[-max_messages:] if self.messages else [] 