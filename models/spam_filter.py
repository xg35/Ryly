from .file_storage_helper import FileStorageHelper
from langchain_community.vectorstores import FAISS # Updated import

class SpamFilter:
    def __init__(self, embeddings_model, blocked_phrases_file="blocked_phrases.txt"):
        """
        Initializes the spam filter, loading blocked phrases from a file
        and building a FAISS vector store (using your embeddings model).
        """
        self.embeddings_model = embeddings_model
        self.blocked_phrases = FileStorageHelper.load_lines(blocked_phrases_file)
        self.blocked_phrases_vector_store = None

        if self.blocked_phrases:
            self.blocked_phrases_vector_store = FAISS.from_texts(
                self.blocked_phrases,
                self.embeddings_model
            )

    def is_blocked_message(self, msg: str, threshold: float = 0.8) -> bool:
        """
        Returns True if `msg` is semantically similar to any blocked phrase
        above the specified threshold. 
        """
        if not self.blocked_phrases or not self.blocked_phrases_vector_store:
            return False

        # Search for the best match among blocked phrases
        results_with_scores = self.blocked_phrases_vector_store.similarity_search_with_score(msg, k=1)
        if not results_with_scores:
            return False

        doc, distance = results_with_scores[0]
        # If distance < (1 - threshold), consider it blocked.
        return (distance < (1.0 - threshold))
