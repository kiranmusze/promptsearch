# PromptSearch

[![PyPI version](https://badge.fury.io/py/promptsearch.svg)](https://badge.fury.io/py/promptsearch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Stop guessing. Start evolving.**

PromptSearch automatically discovers the optimal system prompt for your use case. Give it your target output, and it evolves prompts until they produce exactly what you need.

```
Input: "Contact John Doe at 555-1234"
Target: {"name": "John Doe", "phone": "555-1234"}

PromptSearch runs → Best prompt found in 13 seconds → Score: 1.000 ✓
```

## The Problem

Writing effective prompts is trial and error. You tweak, test, tweak again. Hours lost to manual iteration.

## The Solution

PromptSearch uses **evolutionary hill climbing** to automatically refine prompts:

1. **Generate** → Test current prompt against your input
2. **Score** → Measure output similarity to your target (semantic embeddings)
3. **Mutate** → LLM rewrites the prompt based on failure analysis
4. **Repeat** → Keep improvements, discard regressions

In 3-5 generations, you get a production-ready prompt.

## Installation

```bash
pip install promptsearch
```

## Quick Start

```python
from promptsearch import PromptSearcher

# Define what you want
target = {"name": "John Doe", "phone": "555-1234"}

# Start with a basic prompt
searcher = PromptSearcher(
    target_output=target,
    initial_prompt="Extract contact info from the text."
)

# Let it evolve
result = searcher.optimize(
    train_input="Contact John Doe at 555-1234 for more info.",
    generations=5
)

print(result["best_prompt"])   # Optimized prompt
print(result["best_score"])    # 0.0 - 1.0 similarity
print(result["best_output"])   # Model's output
```

## Multi-Example Training

Train on multiple input/target pairs for robust prompts:

```python
training_data = [
    {"input": "Call Jane at 555-9999", "target": {"name": "Jane", "phone": "555-9999"}},
    {"input": "Reach Bob: 555-0000", "target": {"name": "Bob", "phone": "555-0000"}},
]

result = searcher.optimize(train_input=training_data, generations=5)
```

The optimizer targets the **worst-performing example** each generation, ensuring your prompt handles edge cases.

## Dashboard (Observability)

Every optimization run is logged to SQLite. Visualize the evolution:

```bash
promptsearch-ui
```

**Features:**
- Score evolution chart (Plotly)
- Generation-by-generation prompt inspection
- Side-by-side diff of prompt mutations
- Output vs. target comparison

![Dashboard Preview](https://raw.githubusercontent.com/kiranmusze/promptsearch/main/docs/dashboard.png)

## API Reference

### `PromptSearcher`

```python
PromptSearcher(
    target_output,          # Desired output (any JSON-serializable)
    initial_prompt,         # Starting system prompt
    model="gpt-4o-mini",    # OpenAI model for generation
    scorer_model="all-MiniLM-L6-v2",  # Embedding model for scoring
    db_path="promptsearch.db",        # SQLite log path (None to disable)
    enable_logging=True     # Toggle logging
)
```

### `.optimize()`

```python
result = searcher.optimize(
    train_input,    # Single input or list of {"input": ..., "target": ...}
    generations=5   # Number of evolution cycles
)

# Returns:
# {
#   "best_prompt": str,
#   "best_score": float,       # 0.0 - 1.0
#   "best_output": str | list,
#   "experiment_id": str       # For dashboard lookup
# }
```

## Environment Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                     PromptSearcher                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Generate │───►│  Score   │───►│  Mutate  │──┐           │
│  │ (OpenAI) │    │(Embeddings)   │ (OpenAI) │  │           │
│  └──────────┘    └──────────┘    └──────────┘  │           │
│       ▲                                        │           │
│       └────────────────────────────────────────┘           │
│                    (if improved, keep)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

- **Scorer**: Uses `sentence-transformers` to compute semantic similarity between target and actual output
- **Mutator**: GPT analyzes the failure and rewrites the prompt to fix it
- **Hill Climbing**: Only keeps mutations that improve the average score

## Use Cases

- **Data extraction**: Emails → JSON, logs → structured data
- **Format enforcement**: Ensure consistent output schemas
- **Tone calibration**: Match specific writing styles
- **Few-shot optimization**: Find the best prompt for your examples

## Requirements

- Python 3.8+
- OpenAI API key
- ~100MB for sentence-transformers model (downloaded on first run)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Issues and PRs welcome at [github.com/kiranmusze/promptsearch](https://github.com/kiranmusze/promptsearch).

---

**Built by [Kiran Banakar](mailto:kiranbanakar512@gmail.com)**
