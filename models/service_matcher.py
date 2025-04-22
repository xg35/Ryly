# my_project/models/service_matcher.py

import logging
import numpy as np
from typing import List, Tuple, Optional
from langchain_core.embeddings import Embeddings
from sklearn.metrics.pairwise import cosine_similarity # Need scikit-learn: pip install scikit-learn

logger = logging.getLogger(__name__)

class CustomerServiceMatcher:
    def __init__(self, embeddings_model: Embeddings, file_path: str = "customer_services.txt"):
        self.embeddings_model = embeddings_model
        self.file_path = file_path
        self.service_examples: List[Tuple[str, str]] = [] # List of (type, text)
        self.example_embeddings: Optional[np.ndarray] = None
        self._load_and_embed_examples()

    def _load_and_embed_examples(self):
        """Loads examples from the file and computes their embeddings."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            logger.error(f"Customer service file not found: {self.file_path}. Matcher will be inactive.")
            return
        except Exception as e:
             logger.error(f"Error reading customer service file {self.file_path}: {e}", exc_info=True)
             return

        texts_to_embed = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or ':' not in line:
                if line: logger.warning(f"Skipping invalid line {i+1} in {self.file_path}: '{line}'")
                continue

            parts = line.split(':', 1)
            service_type = parts[0].strip().upper()
            example_text = parts[1].strip()

            if not service_type or not example_text:
                 logger.warning(f"Skipping incomplete line {i+1} in {self.file_path}: '{line}'")
                 continue

            self.service_examples.append((service_type, example_text))
            texts_to_embed.append(example_text)

        if not texts_to_embed:
            logger.warning(f"No valid service examples found in {self.file_path}. Matcher will be inactive.")
            return

        try:
            logger.info(f"Embedding {len(texts_to_embed)} customer service examples...")
            self.example_embeddings = np.array(self.embeddings_model.embed_documents(texts_to_embed))
            # Normalize embeddings for cosine similarity (important if model doesn't guarantee normalized output)
            norms = np.linalg.norm(self.example_embeddings, axis=1, keepdims=True)
            self.example_embeddings = self.example_embeddings / norms # Normalize in place
            logger.info("Customer service examples embedded successfully.")

        except Exception as e:
            logger.error(f"Failed to embed customer service examples: {e}", exc_info=True)
            self.example_embeddings = None # Ensure it's None on failure


    def match(self, message_text: str, threshold: float = 0.8) -> Optional[str]:
        """
        Checks if the message matches any service example above the threshold.
        Returns the service type (e.g., 'ROOM_SERVICE') if matched, else None.
        """
        if self.example_embeddings is None or not self.service_examples:
            # Matcher is inactive (file not found, empty, or embedding failed)
            return None

        try:
            # Embed the incoming message and normalize
            message_embedding = np.array(self.embeddings_model.embed_query(message_text)).reshape(1, -1)
            message_norm = np.linalg.norm(message_embedding)
            if message_norm == 0: return None # Avoid division by zero
            message_embedding_normalized = message_embedding / message_norm


            # Calculate cosine similarities
            similarities = cosine_similarity(message_embedding_normalized, self.example_embeddings)[0] # Get the 1D array of similarities

            # Find the best match
            best_match_index = np.argmax(similarities)
            best_score = similarities[best_match_index]

            logger.debug(f"Customer service match check for '{message_text[:50]}...': Best score={best_score:.4f} (Threshold={threshold})")

            if best_score >= threshold:
                matched_type = self.service_examples[best_match_index][0]
                logger.info(f"Customer service request detected: Type='{matched_type}', Score={best_score:.4f}")
                return matched_type
            else:
                return None

        except Exception as e:
            logger.error(f"Error during customer service matching for message '{message_text[:50]}...': {e}", exc_info=True)
            return None