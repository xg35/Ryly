import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorManager:
    """Handles text chunking, embeddings, and FAISS vector storage"""
    
    def __init__(self, chunk_size=512, chunk_overlap=64):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)  # Matching MiniLM embedding size
        self.doc_store = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def add_document(self, text):
        """Process and store document chunks"""
        chunks = self._chunk_text(text)
        embeddings = self.encoder.encode(chunks)
        self.index.add(np.array(embeddings).astype('float32'))
        self.doc_store.extend(chunks)

    def search(self, query, k=5):
        """Search stored documents"""
        query_embed = self.encoder.encode([query])
        distances, indices = self.index.search(query_embed, k)
        return [self.doc_store[i] for i in indices[0]]

    def _chunk_text(self, text):
        """Custom text splitting without LangChain"""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunks.append(' '.join(words[start:end]))
            start = end - self.chunk_overlap
        return chunks
