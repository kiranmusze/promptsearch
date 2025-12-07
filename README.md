# PromptSearch

[![PyPI version](https://badge.fury.io/py/promptsearch.svg)](https://badge.fury.io/py/promptsearch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automated Reverse Prompt Engineering: Evolve system prompts from target outputs.

## Installation

```bash
pip install promptsearch
```

## Quick Start

```python
from promptsearch import PromptSearcher

# Define your target output (what you want the LLM to produce)
target_output = {
    "phone": "555-1234",
    "name": "John Doe"
}

# Define your initial system prompt
initial_prompt = "Extract information from the text."

# Create the searcher
searcher = PromptSearcher(
    target_output=target_output,
    initial_prompt=initial_prompt
)

# Run optimization
train_input = "Contact John Doe at 555-1234 for more information."
result = searcher.optimize(train_input=train_input, generations=5)

print(f"Best Prompt: {result['best_prompt']}")
print(f"Best Score: {result['best_score']:.3f}")
print(f"Best Output: {result['best_output']}")
```

## How It Works

PromptSearch uses evolutionary hill climbing to optimize system prompts:

1. **Generate**: Test the current prompt with your input
2. **Score**: Compare output to target using semantic similarity
3. **Mutate**: Use an LLM to rewrite the prompt based on failure analysis
4. **Repeat**: Continue until convergence or max generations

## Requirements

- Python 3.8+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## License

MIT License - see LICENSE file for details.

