"""Core prompt optimization loop using evolutionary hill climbing."""

from typing import Any, Optional
from openai import OpenAI
from tqdm import tqdm

from promptsearch.scorers import SemanticScorer
from promptsearch.mutators import PromptMutator


class PromptSearcher:
    """Evolves system prompts to match target outputs using hill climbing."""

    def __init__(
        self,
        target_output: Any,
        initial_prompt: str,
        openai_client: Optional[OpenAI] = None,
        model: str = "gpt-4o-mini",
        scorer_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the prompt searcher.

        Args:
            target_output: The desired target output (can be any type, will be stringified).
            initial_prompt: The starting system prompt to optimize.
            openai_client: OpenAI client instance (creates one if not provided).
            model: OpenAI model to use for generation and mutation.
            scorer_model: Sentence transformer model for scoring.
        """
        self.target_output = target_output
        self.current_prompt = initial_prompt
        self.client = openai_client or OpenAI()
        self.model = model
        self.scorer = SemanticScorer(model_name=scorer_model)
        self.mutator = PromptMutator(client=self.client, model=model)
        self.best_score = 0.0
        self.best_prompt = initial_prompt
        self.best_output = None

    def _generate_output(self, prompt: str, user_input: Any) -> str:
        """Generate output using the current prompt."""
        user_input_str = str(user_input) if not isinstance(user_input, str) else user_input

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input_str},
            ],
            temperature=0.7,
            max_tokens=2000,
        )

        return response.choices[0].message.content.strip()

    def optimize(self, train_input: Any, generations: int = 5) -> dict:
        """
        Optimize the system prompt using hill climbing.

        Args:
            train_input: The input to use for testing prompt variations.
            generations: Number of optimization generations to run.

        Returns:
            Dictionary with 'best_prompt', 'best_score', and 'best_output'.
        """
        # Initial generation
        current_output = self._generate_output(self.current_prompt, train_input)
        current_score = self.scorer.score(self.target_output, current_output)

        self.best_score = current_score
        self.best_prompt = self.current_prompt
        self.best_output = current_output

        # Optimization loop
        with tqdm(total=generations, desc="Optimizing prompt") as pbar:
            for generation in range(generations):
                # Generate output with current prompt
                current_output = self._generate_output(self.current_prompt, train_input)
                current_score = self.scorer.score(self.target_output, current_output)

                # Update best if improved
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_prompt = self.current_prompt
                    self.best_output = current_output
                    pbar.set_postfix({"best_score": f"{self.best_score:.3f}"})

                # Mutate prompt for next generation
                if generation < generations - 1:  # Don't mutate on last iteration
                    self.current_prompt = self.mutator.mutate(
                        current_prompt=self.current_prompt,
                        bad_output=current_output,
                        target_output=str(self.target_output),
                    )

                pbar.update(1)

        return {
            "best_prompt": self.best_prompt,
            "best_score": self.best_score,
            "best_output": self.best_output,
        }

