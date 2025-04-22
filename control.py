import os
import numpy as np
from langchain.vectorstores import FAISS

# ------------------------------------------------------------
# Utility Classes
# ------------------------------------------------------------
class FileStorageHelper:
    @staticmethod
    def load_lines(filepath):
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]

    @staticmethod
    def write_lines(filepath, lines):
        with open(filepath, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

class SpamFilter:
    def __init__(self, embeddings_model, blocked_phrases_file="blocked_phrases.txt"):
        """
        Initializes the spam filter, loading blocked phrases from a file and building
        a FAISS vector store (using the same embeddings model as the rest of your app).
        """
        self.blocked_phrases = FileStorageHelper.load_lines(blocked_phrases_file)
        self.embeddings_model = embeddings_model
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

        Uses a simple L2 distance by default. The vector store's 'score'
        is typically a distance measure where smaller = more similar.
        Adjust threshold logic as needed.
        """
        if not self.blocked_phrases or not self.blocked_phrases_vector_store:
            return False

        # Search for the best match among blocked phrases
        results_with_scores = self.blocked_phrases_vector_store.similarity_search_with_score(msg, k=1)
        if not results_with_scores:
            return False

        # Each entry is (Document, distance)
        doc, distance = results_with_scores[0]

        # If distance < (1 - threshold), it's above the similarity threshold. 
        # This is a simplistic approach; experiment to find the best cutoff.
        return (distance < (1.0 - threshold))
