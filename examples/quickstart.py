"""Quickstart example for PromptSearch with logging enabled."""

import os
from pathlib import Path
from promptsearch import PromptSearcher

# Make sure you have OPENAI_API_KEY set in your environment
# export OPENAI_API_KEY="your-key-here"


def load_env_file(env_path: Path) -> None:
    """Minimal .env reader that sets environment variables from key=value lines."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def main():
    """Run a simple prompt optimization example with automatic logging."""
    
    # Load secrets from secrets.env if available
    env_file = Path(__file__).with_name("secrets.env")
    load_env_file(env_file)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            f"Please set OPENAI_API_KEY in your environment or in {env_file}."
        )
    
    # Define target output (what we want the LLM to produce)
    target_output = {
        "phone": "555-1234",
        "name": "John Doe"
    }
    
    # Initial system prompt (starting point)
    initial_prompt = "Extract information from the text."
    
    # Create the searcher with logging enabled (default)
    # Data will be logged to promptsearch.db in the current directory
    searcher = PromptSearcher(
        target_output=target_output,
        initial_prompt=initial_prompt,
        model="gpt-4o-mini",
        db_path="promptsearch.db",  # SQLite database for logging
        enable_logging=True,  # Enabled by default
    )
    
    # Training input
    train_input = "Contact John Doe at 555-1234 for more information."
    
    print("Starting prompt optimization...")
    print(f"Target output: {target_output}")
    print(f"Initial prompt: {initial_prompt}")
    print(f"Training input: {train_input}")
    print(f"Logging to: promptsearch.db\n")
    
    # Run optimization - all generations are automatically logged
    result = searcher.optimize(train_input=train_input, generations=5)
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"\nBest Prompt:\n{result['best_prompt']}\n")
    print(f"Best Score: {result['best_score']:.3f}")
    print(f"\nBest Output:\n{result['best_output']}\n")
    
    if "experiment_id" in result:
        print(f"Experiment ID: {result['experiment_id']}")
        print("\nView results in the dashboard:")
        print("  promptsearch-ui")
        print("  # or: python -m promptsearch.cli")


if __name__ == "__main__":
    main()
