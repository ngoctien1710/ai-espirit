from typing import Dict, Any
from .embedder import BaseEmbedder
from .vector_store import BaseVectorStore
import logging

class Retriever:
    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)

    def retrieve(self, query: str, top_k: int = 5, filters: Dict = None) -> Dict[str, Any]:
        if not query or not isinstance(query, str):
            return {"documents": [], "metadata": {"status": "ERROR", "error": "Invalid query format"}}
            
        try:
            query_embedding = self.embedder.embed_query(query)
            results = self.vector_store.search(query_embedding, top_k=top_k, filters=filters)
            
            if not results:
                return {"documents": [], "metadata": {"status": "NO_RESULTS", "error": None}}
                
            return {"documents": results, "metadata": {"status": "SUCCESS", "error": None}}
            
        except Exception as e:
            self.logger.error(f"Retrieval system failure: {e}")
            return {"documents": [], "metadata": {"status": "ERROR", "error": str(e)}}