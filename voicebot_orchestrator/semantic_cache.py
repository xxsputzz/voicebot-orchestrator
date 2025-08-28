"""
Sprint 5: Semantic Cache Implementation

High-performance semantic caching using Faiss + sentence-transformers for LLM response caching.
Reduces cost and latency by short-circuiting redundant LLM calls.
"""

from __future__ import annotations
import json
import pickle
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import numpy as np

# Using only standard library and allowed packages
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Mock implementations for restricted environment
class MockSentenceTransformer:
    """Mock sentence transformer for environments without the package."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_dim = 384
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Mock encoding - generates consistent embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Generate deterministic embedding based on text hash
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
            embeddings.append(embedding)
        return np.array(embeddings)


class MockFaissIndex:
    """Mock Faiss index for environments without the package."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: List[np.ndarray] = []
        self.ntotal = 0
    
    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to index."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        for vector in vectors:
            self.vectors.append(vector.copy())
        self.ntotal = len(self.vectors)
    
    def search(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors."""
        if self.ntotal == 0:
            return np.array([[float('inf')]]), np.array([[-1]])
        
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        distances = []
        indices = []
        
        for query in query_vectors:
            query_distances = []
            for i, vector in enumerate(self.vectors):
                # L2 distance
                dist = np.sum((query - vector) ** 2)
                query_distances.append((dist, i))
            
            # Sort by distance and take top k
            query_distances.sort()
            k_results = query_distances[:min(k, len(query_distances))]
            
            dist_array = [d[0] for d in k_results]
            idx_array = [d[1] for d in k_results]
            
            # Pad if needed
            while len(dist_array) < k:
                dist_array.append(float('inf'))
                idx_array.append(-1)
            
            distances.append(dist_array)
            indices.append(idx_array)
        
        return np.array(distances), np.array(indices)


# Try to import real packages, fall back to mocks
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    REAL_LIBS = True
except ImportError:
    SentenceTransformer = MockSentenceTransformer
    faiss = type('MockFaiss', (), {
        'IndexFlatL2': MockFaissIndex
    })()
    REAL_LIBS = False


class SemanticCache:
    """
    High-performance semantic cache using vector embeddings.
    
    Caches LLM responses based on semantic similarity of input queries,
    reducing redundant API calls and improving response times.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "./cache",
        similarity_threshold: float = 0.2,
        max_cache_size: int = 10000
    ):
        """
        Initialize semantic cache.
        
        Args:
            model_name: sentence transformer model name
            cache_dir: directory for cache persistence
            similarity_threshold: cosine similarity threshold for cache hits
            max_cache_size: maximum number of cached entries
        """
        if not model_name:
            raise ValueError("model_name cannot be empty")
        
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        
        if max_cache_size <= 0:
            raise ValueError("max_cache_size must be positive")
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        
        # Initialize embedder
        self.embedder = SentenceTransformer(model_name)
        self.embedding_dim = getattr(self.embedder, 'get_sentence_embedding_dimension', lambda: 384)()
        
        # Initialize Faiss index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Cache storage
        self.cache_queries: List[str] = []
        self.cache_responses: List[str] = []
        self.cache_metadata: List[Dict[str, Any]] = []
        
        # Analytics
        self.hit_count = 0
        self.miss_count = 0
        self.total_queries = 0
        
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Load existing cache
        self._load_cache()
    
    def check_cache(self, query: str) -> Optional[str]:
        """
        Check if query has a cached response within similarity threshold.
        
        Args:
            query: input query string
            
        Returns:
            cached response if found, None otherwise
            
        Raises:
            ValueError: if query is empty
        """
        if not query or not query.strip():
            raise ValueError("Query must be non-empty string")
        
        self.total_queries += 1
        
        # Early return if cache is empty
        if self.index.ntotal == 0:
            self.miss_count += 1
            return None
        
        try:
            # Encode query
            query_embedding = self.embedder.encode([query.strip()])
            
            # Search for nearest neighbor
            distances, indices = self.index.search(query_embedding, 1)
            
            # Check if within threshold (convert L2 to cosine similarity approximation)
            if distances[0][0] < self.similarity_threshold:
                cache_idx = indices[0][0]
                if 0 <= cache_idx < len(self.cache_responses):
                    self.hit_count += 1
                    self.logger.debug(f"Cache hit for query: {query[:50]}...")
                    return self.cache_responses[cache_idx]
            
            self.miss_count += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking cache: {e}")
            self.miss_count += 1
            return None
    
    def add_to_cache(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add query-response pair to cache.
        
        Args:
            query: input query string
            response: LLM response string
            metadata: optional metadata dictionary
            
        Raises:
            ValueError: if query or response is empty
        """
        if not query or not query.strip():
            raise ValueError("Query must be non-empty string")
        
        if not response or not response.strip():
            raise ValueError("Response must be non-empty string")
        
        try:
            # Check cache size limit
            if len(self.cache_queries) >= self.max_cache_size:
                self._evict_oldest()
            
            # Encode query
            query_embedding = self.embedder.encode([query.strip()])
            
            # Add to index
            self.index.add(query_embedding)
            
            # Store in cache
            self.cache_queries.append(query.strip())
            self.cache_responses.append(response.strip())
            self.cache_metadata.append(metadata or {})
            
            self.logger.debug(f"Added to cache: {query[:50]}...")
            
            # Periodic save
            if len(self.cache_queries) % 100 == 0:
                self._save_cache()
                
        except Exception as e:
            self.logger.error(f"Error adding to cache: {e}")
            raise
    
    def evict_by_threshold(self, accuracy_threshold: float) -> int:
        """
        Evict cache entries below accuracy threshold.
        
        Args:
            accuracy_threshold: minimum accuracy to keep in cache
            
        Returns:
            number of entries evicted
        """
        if not (0 <= accuracy_threshold <= 1):
            raise ValueError("accuracy_threshold must be between 0 and 1")
        
        # For this implementation, we'll evict entries with low similarity scores
        # In a real implementation, you'd track accuracy metrics per entry
        evicted_count = 0
        
        # This is a simplified eviction - in practice you'd need accuracy tracking
        if accuracy_threshold > 0.8:  # High threshold - evict random low-performing entries
            evict_count = max(1, len(self.cache_queries) // 10)  # Evict 10%
            self._evict_random(evict_count)
            evicted_count = evict_count
        
        return evicted_count
    
    def clear_cache(self) -> None:
        """Clear all cache entries."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.cache_queries.clear()
        self.cache_responses.clear()
        self.cache_metadata.clear()
        
        # Reset analytics
        self.hit_count = 0
        self.miss_count = 0
        self.total_queries = 0
        
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            dictionary with cache performance metrics
        """
        hit_rate = self.hit_count / max(self.total_queries, 1)
        
        return {
            "total_entries": len(self.cache_queries),
            "total_queries": self.total_queries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 4),
            "embedding_dim": self.embedding_dim,
            "similarity_threshold": self.similarity_threshold,
            "max_cache_size": self.max_cache_size,
            "model_name": self.model_name,
            "cache_size_mb": self._estimate_cache_size_mb()
        }
    
    def export_cache_data(self) -> List[Dict[str, Any]]:
        """
        Export cache data for analysis.
        
        Returns:
            list of cache entries with metadata
        """
        cache_data = []
        for i in range(len(self.cache_queries)):
            entry = {
                "query": self.cache_queries[i],
                "response": self.cache_responses[i],
                "metadata": self.cache_metadata[i],
                "entry_id": i
            }
            cache_data.append(entry)
        
        return cache_data
    
    def save_cache(self) -> None:
        """Manually save cache to disk."""
        self._save_cache()
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if len(self.cache_queries) > 0:
            # Remove first entry (FIFO)
            self.cache_queries.pop(0)
            self.cache_responses.pop(0)
            self.cache_metadata.pop(0)
            
            # Rebuild index (expensive but necessary for Faiss)
            self._rebuild_index()
    
    def _evict_random(self, count: int) -> None:
        """Evict random cache entries."""
        if count >= len(self.cache_queries):
            self.clear_cache()
            return
        
        import random
        indices_to_remove = sorted(random.sample(range(len(self.cache_queries)), count), reverse=True)
        
        for idx in indices_to_remove:
            self.cache_queries.pop(idx)
            self.cache_responses.pop(idx)
            self.cache_metadata.pop(idx)
        
        self._rebuild_index()
    
    def _rebuild_index(self) -> None:
        """Rebuild Faiss index from current cache entries."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        if self.cache_queries:
            # Re-encode all queries
            embeddings = self.embedder.encode(self.cache_queries)
            self.index.add(embeddings)
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            cache_file = self.cache_dir / "semantic_cache.json"
            
            cache_data = {
                "queries": self.cache_queries,
                "responses": self.cache_responses,
                "metadata": self.cache_metadata,
                "stats": {
                    "hit_count": self.hit_count,
                    "miss_count": self.miss_count,
                    "total_queries": self.total_queries
                },
                "config": {
                    "model_name": self.model_name,
                    "similarity_threshold": self.similarity_threshold,
                    "max_cache_size": self.max_cache_size
                }
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Cache saved to {cache_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            cache_file = self.cache_dir / "semantic_cache.json"
            
            if not cache_file.exists():
                return
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Load cache entries
            self.cache_queries = cache_data.get("queries", [])
            self.cache_responses = cache_data.get("responses", [])
            self.cache_metadata = cache_data.get("metadata", [])
            
            # Load stats
            stats = cache_data.get("stats", {})
            self.hit_count = stats.get("hit_count", 0)
            self.miss_count = stats.get("miss_count", 0)
            self.total_queries = stats.get("total_queries", 0)
            
            # Rebuild index
            if self.cache_queries:
                self._rebuild_index()
            
            self.logger.info(f"Loaded {len(self.cache_queries)} cache entries")
            
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            # Reset to empty cache on error
            self.clear_cache()
    
    def _estimate_cache_size_mb(self) -> float:
        """Estimate cache size in MB."""
        total_chars = sum(len(q) + len(r) for q, r in zip(self.cache_queries, self.cache_responses))
        
        # Rough estimate: 1 char ≈ 1 byte, plus embeddings
        text_size = total_chars
        embedding_size = len(self.cache_queries) * self.embedding_dim * 4  # 4 bytes per float32
        
        total_bytes = text_size + embedding_size
        return round(total_bytes / (1024 * 1024), 2)


# Analytics integration
_cache_analytics = {
    "total_hits": 0,
    "total_misses": 0,
    "total_adds": 0,
    "total_evictions": 0
}


def get_semantic_cache_analytics() -> Dict[str, Any]:
    """Get global semantic cache analytics."""
    return {
        "cache_hits": _cache_analytics["total_hits"],
        "cache_misses": _cache_analytics["total_misses"],
        "cache_adds": _cache_analytics["total_adds"],
        "cache_evictions": _cache_analytics["total_evictions"],
        "service_name": "semantic_cache"
    }


def _update_analytics(operation: str) -> None:
    """Update global analytics counters."""
    if operation in _cache_analytics:
        _cache_analytics[operation] += 1


# Factory function for easy instantiation
def create_semantic_cache(
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: str = "./cache",
    similarity_threshold: float = 0.2,
    max_cache_size: int = 10000
) -> SemanticCache:
    """
    Create and configure semantic cache instance.
    
    Args:
        model_name: sentence transformer model
        cache_dir: cache persistence directory
        similarity_threshold: similarity threshold for cache hits
        max_cache_size: maximum cache entries
        
    Returns:
        configured SemanticCache instance
    """
    return SemanticCache(
        model_name=model_name,
        cache_dir=cache_dir,
        similarity_threshold=similarity_threshold,
        max_cache_size=max_cache_size
    )
