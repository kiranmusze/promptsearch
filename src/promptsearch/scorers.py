"""Semantic similarity scoring for prompt optimization."""

from typing import Any
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticScorer:
    """Scores similarity between target and actual outputs using sentence transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic scorer.

        Args:
            model_name: Name of the sentence transformer model to use.
        """
        self.model = SentenceTransformer(model_name)

    def score(self, target: Any, actual: Any) -> float:
        """
        Compute cosine similarity between target and actual outputs.

        Args:
            target: The target output (will be converted to string).
            actual: The actual output (will be converted to string).

        Returns:
            Cosine similarity score between 0.0 and 1.0.
        """
        # Convert to strings if not already
        target_str = str(target) if not isinstance(target, str) else target
        actual_str = str(actual) if not isinstance(actual, str) else actual

        # Generate embeddings
        target_embedding = self.model.encode(target_str, convert_to_numpy=True)
        actual_embedding = self.model.encode(actual_str, convert_to_numpy=True)

        # Compute cosine similarity
        similarity = np.dot(target_embedding, actual_embedding) / (
            np.linalg.norm(target_embedding) * np.linalg.norm(actual_embedding)
        )

        return float(similarity)

