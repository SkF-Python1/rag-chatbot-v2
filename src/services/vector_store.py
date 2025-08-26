"""
FAISS-based vector store service for document embeddings and similarity search.
"""

import os
import pickle
import numpy as np
import faiss
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# OpenAI for embeddings
import openai
from openai import OpenAI

# Local imports
from ..models.document import DocumentChunk, DocumentSearchResult
from ..core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.openai_embedding_model
        self.index_path = settings.faiss_index_path
        self.metadata_path = f"{self.index_path}_metadata.pkl"
        
        # FAISS index and metadata
        self.index: Optional[faiss.Index] = None
        self.chunk_metadata: Dict[int, DocumentChunk] = {}
        self.dimension: int = 1536  # OpenAI text-embedding-ada-002 dimension
        
        # Performance settings
        self.search_k = settings.vector_search_k
        
        # Initialize or load existing index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index or load existing one."""
        try:
            if os.path.exists(f"{self.index_path}.index") and os.path.exists(self.metadata_path):
                self._load_index()
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                self._create_new_index()
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Use HNSW index for better performance with large datasets
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 64
        self.chunk_metadata = {}
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        try:
            self.index = faiss.read_index(f"{self.index_path}.index")
            
            with open(self.metadata_path, 'rb') as f:
                self.chunk_metadata = pickle.load(f)
            
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{self.index_path}.index")
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.chunk_metadata, f)
            
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    async def add_document_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not chunks:
                return True
            
            # Generate embeddings for all chunks
            embeddings = await self._generate_embeddings([chunk.content for chunk in chunks])
            
            if len(embeddings) != len(chunks):
                logger.error("Mismatch between embeddings and chunks count")
                return False
            
            # Add to FAISS index
            embeddings_array = np.array(embeddings).astype('float32')
            start_id = self.index.ntotal
            
            self.index.add(embeddings_array)
            
            # Store chunk metadata
            for i, chunk in enumerate(chunks):
                chunk_id = start_id + i
                self.chunk_metadata[chunk_id] = chunk
            
            # Save index
            self._save_index()
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {str(e)}")
            return False
    
    async def search_similar_chunks(
        self, 
        query: str, 
        k: Optional[int] = None,
        filter_document_ids: Optional[List[str]] = None
    ) -> List[DocumentSearchResult]:
        """
        Search for similar document chunks.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_document_ids: Optional list of document IDs to filter by
            
        Returns:
            List of search results with chunks and similarity scores
        """
        try:
            if k is None:
                k = self.search_k
            
            if self.index.ntotal == 0:
                logger.warning("Vector store is empty")
                return []
            
            # Generate query embedding
            query_embeddings = await self._generate_embeddings([query])
            if not query_embeddings:
                return []
            
            query_vector = np.array(query_embeddings[0]).astype('float32').reshape(1, -1)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_vector, min(k * 2, self.index.ntotal))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                if idx in self.chunk_metadata:
                    chunk = self.chunk_metadata[idx]
                    
                    # Apply document ID filter if specified
                    if filter_document_ids and chunk.document_id not in filter_document_ids:
                        continue
                    
                    # Convert FAISS distance to similarity score (0-1 range)
                    similarity_score = float(1 / (1 + score))
                    
                    result = DocumentSearchResult(
                        chunk=chunk,
                        score=similarity_score,
                        document_metadata=None  # Will be populated by the service layer
                    )
                    results.append(result)
                    
                    if len(results) >= k:
                        break
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            # Batch texts to avoid API limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Call OpenAI embeddings API
                response = await asyncio.to_thread(
                    self.client.embeddings.create,
                    input=batch_texts,
                    model=self.embedding_model
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                if len(texts) > batch_size:
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    async def remove_document(self, document_id: str) -> bool:
        """
        Remove all chunks for a document from the vector store.
        Note: FAISS doesn't support efficient deletion, so this rebuilds the index.
        
        Args:
            document_id: Document ID to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find chunks to keep
            chunks_to_keep = []
            for chunk in self.chunk_metadata.values():
                if chunk.document_id != document_id:
                    chunks_to_keep.append(chunk)
            
            if len(chunks_to_keep) == len(self.chunk_metadata):
                logger.warning(f"No chunks found for document {document_id}")
                return True
            
            # Rebuild index with remaining chunks
            logger.info(f"Rebuilding index to remove document {document_id}")
            self._create_new_index()
            
            if chunks_to_keep:
                success = await self.add_document_chunks(chunks_to_keep)
                if not success:
                    logger.error("Failed to rebuild index after document removal")
                    return False
            
            logger.info(f"Successfully removed document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document {document_id}: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__ if self.index else None,
            "total_chunks": len(self.chunk_metadata),
            "index_file_exists": os.path.exists(f"{self.index_path}.index"),
            "metadata_file_exists": os.path.exists(self.metadata_path)
        }
    
    async def optimize_index(self):
        """Optimize the FAISS index for better search performance."""
        try:
            if self.index.ntotal == 0:
                return
            
            logger.info("Optimizing FAISS index...")
            
            # For HNSW, we can adjust search parameters
            if hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = min(128, max(64, self.index.ntotal // 100))
            
            self._save_index()
            logger.info("Index optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing index: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vector store."""
        try:
            stats = self.get_stats()
            
            # Test embedding generation
            test_embedding = await self._generate_embeddings(["test text"])
            embedding_test_passed = len(test_embedding) == 1 and len(test_embedding[0]) == self.dimension
            
            # Test search if index has data
            search_test_passed = True
            if self.index.ntotal > 0:
                try:
                    test_results = await self.search_similar_chunks("test query", k=1)
                    search_test_passed = isinstance(test_results, list)
                except:
                    search_test_passed = False
            
            return {
                "status": "healthy" if embedding_test_passed and search_test_passed else "unhealthy",
                "embedding_test": embedding_test_passed,
                "search_test": search_test_passed,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "stats": self.get_stats()
            } 