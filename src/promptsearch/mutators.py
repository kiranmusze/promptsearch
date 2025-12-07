"""LLM-based prompt mutation for evolutionary optimization."""

from typing import Optional
from openai import OpenAI


class PromptMutator:
    """Mutates system prompts using LLM-based rewriting."""

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        """
        Initialize the prompt mutator.

        Args:
            client: OpenAI client instance.
            model: Model to use for mutation (default: gpt-4o-mini).
        """
        self.client = client
        self.model = model

    def mutate(
        self,
        current_prompt: str,
        bad_output: str,
        target_output: str,
    ) -> str:
        """
        Generate a mutated version of the system prompt based on failure analysis.

        Args:
            current_prompt: The current system prompt to mutate.
            bad_output: The output that didn't match the target.
            target_output: The desired target output.

        Returns:
            A new mutated system prompt.
        """
        mutation_prompt = f"""You are a prompt engineering expert. Your task is to improve a system prompt based on a failure analysis.

CURRENT SYSTEM PROMPT:
{current_prompt}

ACTUAL OUTPUT (that failed):
{bad_output}

TARGET OUTPUT (desired):
{target_output}

Analyze why the current prompt produced the wrong output. Then rewrite the system prompt to better guide the model toward producing outputs like the target. 

Your response should contain ONLY the improved system prompt text, without any explanation or markdown formatting. Just the prompt itself."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a prompt engineering expert. Return only the improved prompt text, no explanations.",
                },
                {"role": "user", "content": mutation_prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        mutated_prompt = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if mutated_prompt.startswith("```"):
            lines = mutated_prompt.split("\n")
            mutated_prompt = "\n".join(lines[1:-1]) if len(lines) > 2 else mutated_prompt

        return mutated_prompt

