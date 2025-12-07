"""Quickstart example for PromptSearch."""

import os
from promptsearch import PromptSearcher

# Make sure you have OPENAI_API_KEY set in your environment
# export OPENAI_API_KEY="your-key-here"

def main():
    """Run a simple prompt optimization example."""
    
    # Define target output (what we want the LLM to produce)
    target_output = {
        "phone": "555-1234",
        "name": "John Doe"
    }
    
    # Initial system prompt (starting point)
    initial_prompt = "Extract information from the text."
    
    # Create the searcher
    searcher = PromptSearcher(
        target_output=target_output,
        initial_prompt=initial_prompt,
        model="gpt-4o-mini"  # Use a cheaper model for testing
    )
    
    # Training input
    train_input = "Contact John Doe at 555-1234 for more information."
    
    print("Starting prompt optimization...")
    print(f"Target output: {target_output}")
    print(f"Initial prompt: {initial_prompt}")
    print(f"Training input: {train_input}\n")
    
    # Run optimization
    result = searcher.optimize(train_input=train_input, generations=5)
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"\nBest Prompt:\n{result['best_prompt']}\n")
    print(f"Best Score: {result['best_score']:.3f}")
    print(f"\nBest Output:\n{result['best_output']}\n")


if __name__ == "__main__":
    main()

