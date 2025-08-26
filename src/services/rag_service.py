"""
RAG (Retrieval-Augmented Generation) service using LangChain and OpenAI GPT-4.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import logging

# LangChain imports
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.callbacks import AsyncCallbackManager
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# OpenAI direct import for faster responses
from openai import OpenAI

# Local imports
from .vector_store import VectorStore
from ..models.chat import ChatRequest, ChatResponse, ChatHistory, ChatMessage
from ..models.document import DocumentSearchResult
from ..core.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service for context-aware question answering."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # LangChain LLM
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.1,
            max_tokens=settings.max_tokens_per_response,
            timeout=settings.response_timeout,
            api_key=settings.openai_api_key,
            streaming=True  # Enable streaming for faster perceived response
        )
        
        # RAG prompt template
        self.rag_prompt = self._create_rag_prompt()
        
        # Conversation histories (in production, use Redis or database)
        self.conversations: Dict[str, ChatHistory] = {}
    
    def _create_rag_prompt(self) -> ChatPromptTemplate:
        """Create the RAG prompt template for context-aware responses."""
        system_prompt = """You are an intelligent assistant for a knowledge base system. 
        You provide accurate, helpful answers based on the provided context from documents.

        INSTRUCTIONS:
        1. Use the provided context to answer the user's question accurately
        2. If the context doesn't contain enough information, say so clearly
        3. Cite specific parts of the context when possible
        4. Be concise but comprehensive in your responses
        5. Maintain a professional and helpful tone
        6. If asked about something not in the context, politely redirect to the available information

        CONTEXT:
        {context}

        CONVERSATION HISTORY:
        {history}
        """
        
        human_prompt = "Question: {question}"
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request using RAG pipeline.
        
        Args:
            request: Chat request with message and optional parameters
            
        Returns:
            Chat response with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Get or create conversation
            conversation = self._get_or_create_conversation(request.conversation_id)
            
            # Add user message to history
            conversation.add_message("user", request.message)
            
            # Retrieve relevant context if enabled
            context_chunks = []
            if request.use_context:
                context_chunks = await self._retrieve_context(
                    request.message,
                    conversation_history=conversation.get_context_messages()
                )
            
            # Generate response using RAG
            response_text, token_count = await self._generate_response(
                question=request.message,
                context_chunks=context_chunks,
                conversation_history=conversation.get_context_messages(),
                max_tokens=request.max_tokens
            )
            
            # Add assistant response to history
            conversation.add_message("assistant", response_text)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Prepare source information
            sources = [
                {
                    "document_id": chunk.chunk.document_id,
                    "chunk_id": chunk.chunk.id,
                    "content": chunk.chunk.content[:200] + "..." if len(chunk.chunk.content) > 200 else chunk.chunk.content,
                    "score": chunk.score,
                    "page_number": chunk.chunk.page_number,
                    "section": chunk.chunk.section
                }
                for chunk in context_chunks
            ]
            
            logger.info(f"RAG response generated in {response_time:.3f}s with {len(sources)} sources")
            
            return ChatResponse(
                response=response_text,
                conversation_id=conversation.conversation_id,
                sources=sources,
                response_time=response_time,
                token_count=token_count
            )
            
        except Exception as e:
            logger.error(f"Error in RAG chat: {str(e)}")
            response_time = time.time() - start_time
            
            return ChatResponse(
                response="I apologize, but I encountered an error while processing your request. Please try again.",
                conversation_id=request.conversation_id or "error",
                sources=[],
                response_time=response_time,
                token_count=0
            )
    
    async def _retrieve_context(
        self, 
        query: str, 
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> List[DocumentSearchResult]:
        """
        Retrieve relevant context chunks for the query.
        
        Args:
            query: User query
            conversation_history: Recent conversation messages for context
            
        Returns:
            List of relevant document chunks
        """
        try:
            # Enhance query with conversation context
            enhanced_query = await self._enhance_query_with_history(query, conversation_history)
            
            # Search vector store
            search_results = await self.vector_store.search_similar_chunks(
                query=enhanced_query,
                k=settings.vector_search_k
            )
            
            # Filter results by relevance threshold
            filtered_results = [
                result for result in search_results 
                if result.score > 0.5  # Adjust threshold as needed
            ]
            
            logger.info(f"Retrieved {len(filtered_results)} relevant chunks for query")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    async def _enhance_query_with_history(
        self, 
        query: str, 
        history: Optional[List[ChatMessage]] = None
    ) -> str:
        """
        Enhance the query with conversation context for better retrieval.
        
        Args:
            query: Original user query
            history: Conversation history
            
        Returns:
            Enhanced query string
        """
        if not history or len(history) == 0:
            return query
        
        try:
            # Get last few messages for context
            recent_messages = history[-3:] if len(history) > 3 else history
            
            # Create context from recent messages
            context_parts = []
            for msg in recent_messages:
                if msg.role == "user":
                    context_parts.append(f"Previous question: {msg.content}")
                elif msg.role == "assistant":
                    context_parts.append(f"Previous answer: {msg.content[:100]}...")
            
            if context_parts:
                enhanced_query = f"{query}\n\nContext from conversation:\n" + "\n".join(context_parts)
                return enhanced_query
            
        except Exception as e:
            logger.warning(f"Error enhancing query with history: {str(e)}")
        
        return query
    
    async def _generate_response(
        self,
        question: str,
        context_chunks: List[DocumentSearchResult],
        conversation_history: Optional[List[ChatMessage]] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, int]:
        """
        Generate response using LangChain and OpenAI GPT-4.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            conversation_history: Conversation history
            max_tokens: Maximum tokens in response
            
        Returns:
            Tuple of (response_text, token_count)
        """
        try:
            # Prepare context from chunks
            context = self._format_context(context_chunks)
            
            # Prepare conversation history
            history = self._format_history(conversation_history)
            
            # Use direct OpenAI API for faster response
            if max_tokens is None:
                max_tokens = settings.max_tokens_per_response
            
            # Create messages for OpenAI API
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an intelligent assistant for a knowledge base system. 
                    You provide accurate, helpful answers based on the provided context from documents.

                    INSTRUCTIONS:
                    1. Use the provided context to answer the user's question accurately
                    2. If the context doesn't contain enough information, say so clearly
                    3. Cite specific parts of the context when possible
                    4. Be concise but comprehensive in your responses
                    5. Maintain a professional and helpful tone
                    6. If asked about something not in the context, politely redirect to the available information

                    CONTEXT:
                    {context}

                    CONVERSATION HISTORY:
                    {history}
                    """
                },
                {
                    "role": "user",
                    "content": f"Question: {question}"
                }
            ]
            
            # Call OpenAI API directly for speed
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=settings.openai_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                timeout=settings.response_timeout
            )
            
            response_text = response.choices[0].message.content
            token_count = response.usage.completion_tokens if response.usage else 0
            
            return response_text, token_count
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I couldn't generate a response at this time. Please try again.", 0
    
    def _format_context(self, context_chunks: List[DocumentSearchResult]) -> str:
        """Format context chunks for the prompt."""
        if not context_chunks:
            return "No relevant context found in the knowledge base."
        
        formatted_chunks = []
        for i, chunk_result in enumerate(context_chunks, 1):
            chunk = chunk_result.chunk
            score = chunk_result.score
            
            formatted_chunk = f"""
            [Source {i}] (Relevance: {score:.2f})
            Content: {chunk.content}
            {f"Page: {chunk.page_number}" if chunk.page_number else ""}
            {f"Section: {chunk.section}" if chunk.section else ""}
            """
            formatted_chunks.append(formatted_chunk.strip())
        
        return "\n\n".join(formatted_chunks)
    
    def _format_history(self, history: Optional[List[ChatMessage]]) -> str:
        """Format conversation history for the prompt."""
        if not history:
            return "No previous conversation."
        
        formatted_messages = []
        for msg in history[-5:]:  # Only include last 5 messages
            role = "User" if msg.role == "user" else "Assistant"
            formatted_messages.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted_messages)
    
    def _get_or_create_conversation(self, conversation_id: Optional[str]) -> ChatHistory:
        """Get existing conversation or create a new one."""
        if conversation_id and conversation_id in self.conversations:
            return self.conversations[conversation_id]
        
        # Create new conversation
        import uuid
        new_id = conversation_id or str(uuid.uuid4())
        conversation = ChatHistory(conversation_id=new_id)
        self.conversations[new_id] = conversation
        
        return conversation
    
    async def get_conversation_history(self, conversation_id: str) -> Optional[ChatHistory]:
        """Get conversation history by ID."""
        return self.conversations.get(conversation_id)
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the RAG service."""
        try:
            # Test vector store
            vector_health = await self.vector_store.health_check()
            
            # Test OpenAI connection
            openai_test = await self._test_openai_connection()
            
            # Test full pipeline with a simple query
            pipeline_test = await self._test_rag_pipeline()
            
            return {
                "status": "healthy" if all([
                    vector_health.get("status") == "healthy",
                    openai_test,
                    pipeline_test
                ]) else "unhealthy",
                "vector_store": vector_health,
                "openai_connection": openai_test,
                "pipeline_test": pipeline_test,
                "active_conversations": len(self.conversations)
            }
            
        except Exception as e:
            logger.error(f"RAG service health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _test_openai_connection(self) -> bool:
        """Test OpenAI API connection."""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=settings.openai_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                timeout=5
            )
            return bool(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {str(e)}")
            return False
    
    async def _test_rag_pipeline(self) -> bool:
        """Test the full RAG pipeline."""
        try:
            test_request = ChatRequest(
                message="What is this system?",
                use_context=True
            )
            response = await self.chat(test_request)
            return bool(response.response) and response.response_time < 5.0
        except Exception as e:
            logger.error(f"RAG pipeline test failed: {str(e)}")
            return False 