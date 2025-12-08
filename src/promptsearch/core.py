"""Core prompt optimization loop using evolutionary hill climbing."""

from typing import Any, List, Optional, Sequence, Tuple, Union
from openai import OpenAI
from tqdm import tqdm

from promptsearch.scorers import SemanticScorer
from promptsearch.mutators import PromptMutator
from promptsearch.logger import SQLiteLogger


class PromptSearcher:
    """Evolves system prompts to match target outputs using hill climbing."""

    def __init__(
        self,
        target_output: Any,
        initial_prompt: str,
        openai_client: Optional[OpenAI] = None,
        model: str = "gpt-4o-mini",
        scorer_model: str = "all-MiniLM-L6-v2",
        db_path: Optional[str] = "promptsearch.db",
        enable_logging: bool = True,
    ):
        """
        Initialize the prompt searcher.

        Args:
            target_output: The desired target output (can be any type, will be stringified).
            initial_prompt: The starting system prompt to optimize.
            openai_client: OpenAI client instance (creates one if not provided).
            model: OpenAI model to use for generation and mutation.
            scorer_model: Sentence transformer model for scoring.
            db_path: Path to SQLite database for logging. Set to None to disable.
            enable_logging: Whether to log experiments to the database.
        """
        self.target_output = target_output
        self.initial_prompt = initial_prompt
        self.current_prompt = initial_prompt
        self.client = openai_client or OpenAI()
        self.model = model
        self.scorer = SemanticScorer(model_name=scorer_model)
        self.mutator = PromptMutator(client=self.client, model=model)
        self.best_score = 0.0
        self.best_prompt = initial_prompt
        self.best_output = None

        # Logging
        self.enable_logging = enable_logging and db_path is not None
        self.logger: Optional[SQLiteLogger] = None
        if self.enable_logging:
            self.logger = SQLiteLogger(db_path=db_path)

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
            train_input: Either a single input (string/any) or a list of items.
                For multiple examples, provide a list of dicts/tuples:
                - {"input": <text>, "target": <desired_output>}
                - (<input>, <target>)
            generations: Number of optimization generations to run.

        Returns:
            Dictionary with:
                - 'best_prompt'
                - 'best_score' (average over dataset if multiple)
                - 'best_output' (single) or 'best_outputs' (list of tuples)
                - 'experiment_id' (if logging enabled)
        """

        def _normalize_dataset(data: Any) -> List[Tuple[Any, Any]]:
            # If list/sequence: expect items as (input, target) or dicts with keys.
            if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
                normalized = []
                for item in data:
                    if isinstance(item, dict):
                        inp = item.get("input") or item.get("email") or item.get("text") or item.get("prompt")
                        tgt = item.get("target") or item.get("output")
                    elif isinstance(item, tuple) and len(item) == 2:
                        inp, tgt = item
                    else:
                        continue
                    if inp is not None and tgt is not None:
                        normalized.append((inp, tgt))
                if not normalized:
                    raise ValueError("No valid (input, target) pairs found in train_input.")
                return normalized

            # Single item fallback: use self.target_output
            if self.target_output is None:
                raise ValueError("target_output is required when train_input is a single item.")
            return [(data, self.target_output)]

        def _evaluate_prompt(prompt: str, dataset: List[Tuple[Any, Any]]) -> Tuple[float, List[Tuple[Any, Any, Any, float]]]:
            """Returns (avg_score, records) where records are (input, output, target, score)."""
            records = []
            for inp, tgt in dataset:
                out = self._generate_output(prompt, inp)
                score = self.scorer.score(tgt, out)
                records.append((inp, out, tgt, score))
            avg_score = sum(r[3] for r in records) / len(records)
            return avg_score, records

        dataset = _normalize_dataset(train_input)

        # Start experiment logging
        experiment_id = None
        if self.logger:
            # Use first target as representative for experiment metadata
            representative_target = dataset[0][1] if len(dataset) == 1 else [d[1] for d in dataset]
            experiment_id = self.logger.start_experiment(
                initial_prompt=self.initial_prompt,
                target_output=representative_target,
                model=self.model,
                generations=generations,
            )

        # Initial evaluation
        current_score, current_records = _evaluate_prompt(self.current_prompt, dataset)
        self.best_score = current_score
        self.best_prompt = self.current_prompt
        self.best_output = current_records if len(dataset) > 1 else current_records[0][1]

        # Log initial generation (step 0)
        if self.logger and experiment_id:
            for inp, out, tgt, score in current_records:
                self.logger.log_generation(
                    experiment_id=experiment_id,
                    step_num=0,
                    prompt_text=self.current_prompt,
                    input_text=str(inp),
                    output_text=str(out),
                    target_text=str(tgt),
                    score=score,
                    is_best=(score == max(r[3] for r in current_records)),
                )

        with tqdm(total=generations, desc="Optimizing prompt") as pbar:
            for generation in range(generations):
                # Evaluate current prompt
                current_score, current_records = _evaluate_prompt(self.current_prompt, dataset)

                # Log this generation
                if self.logger and experiment_id:
                    for inp, out, tgt, score in current_records:
                        self.logger.log_generation(
                            experiment_id=experiment_id,
                            step_num=generation + 1,
                            prompt_text=self.current_prompt,
                            input_text=str(inp),
                            output_text=str(out),
                            target_text=str(tgt),
                            score=score,
                            is_best=(current_score > self.best_score),
                        )

                # Update best if improved
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_prompt = self.current_prompt
                    self.best_output = current_records if len(dataset) > 1 else current_records[0][1]
                    pbar.set_postfix({"best_score": f"{self.best_score:.3f}"})

                # Mutate prompt for next generation (skip on last)
                if generation < generations - 1:
                    # Pick the lowest-scoring example to guide mutation
                    worst = min(current_records, key=lambda r: r[3])
                    worst_input, worst_output, worst_target, _ = worst

                    self.current_prompt = self.mutator.mutate(
                        current_prompt=self.current_prompt,
                        bad_output=str(worst_output),
                        target_output=str(worst_target),
                    )

                pbar.update(1)

        # Complete experiment logging
        if self.logger and experiment_id:
            self.logger.complete_experiment(
                experiment_id=experiment_id,
                best_score=self.best_score,
                best_prompt=self.best_prompt,
            )

        result = {
            "best_prompt": self.best_prompt,
            "best_score": self.best_score,
            "best_output": self.best_output,
        }

        if experiment_id:
            result["experiment_id"] = experiment_id

        return result
