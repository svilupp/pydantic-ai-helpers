# Examples

This directory contains example scripts demonstrating the key features of pydantic-ai-helpers.

## Basic Usage Examples

### `basic_usage.py`
Demonstrates the core History functionality for extracting information from PydanticAI conversations:
- User, AI, and system message access
- Tool call and return extraction
- Token usage tracking
- Media content handling

### `media_and_system_prompts.py`
Shows how to work with media content and system prompts:
- Image, audio, video, and document extraction
- System prompt analysis
- Media filtering by type and storage method

### `advanced_patterns.py`
Advanced usage patterns including:
- Multi-turn conversation analysis
- Conversation persistence and restoration
- Cost tracking and token usage monitoring
- Streaming response handling

## Fuzzy Matching Examples (NEW!)

### `fuzzy_quick_start.py` ⭐ **Start Here**
A quick introduction to fuzzy matching for AI evaluation:
- Basic string comparisons with fuzzy matching
- List evaluation with fuzzy scores
- Field-to-field evaluation examples
- Category validation
- Real AI evaluation scenarios

**Perfect for getting started with fuzzy matching!**

### `fuzzy_matching_demo.py`
Comprehensive demonstration of fuzzy matching capabilities:
- Algorithm comparison (ratio, partial_ratio, token_sort_ratio, token_set_ratio)
- Threshold sensitivity analysis
- Real-world AI evaluation scenarios
- Advanced configuration options
- Performance comparisons

### `ai_evaluation_pipeline.py`
Production-ready AI evaluation pipeline example:
- Complete evaluation pipeline setup
- Multiple evaluator coordination
- Summary statistics and reporting
- Fuzzy vs exact matching comparison
- Best practices for AI content evaluation

## Running the Examples

All examples are self-contained Python scripts:

```bash
# Basic functionality
python examples/basic_usage.py
python examples/media_and_system_prompts.py
python examples/advanced_patterns.py

# Fuzzy matching (NEW!)
python examples/fuzzy_quick_start.py         # Start here!
python examples/fuzzy_matching_demo.py
python examples/ai_evaluation_pipeline.py
```

## Key Features Demonstrated

### History Extraction
- Message parsing and filtering
- Tool call/return analysis
- Token usage and cost tracking
- Media content extraction

### Fuzzy Matching Evaluation
- **Default fuzzy matching** with 0.85 threshold
- **Multiple algorithms** for different use cases
- **Normalization** before fuzzy matching
- **List evaluations** with fuzzy scores
- **Category validation** with fuzzy fallback
- **Real-world AI evaluation** scenarios

### Best Practices
- Type-safe API usage
- Error handling patterns
- Performance optimization
- Evaluation pipeline design

## Dependencies

Examples require:
- `pydantic-ai-helpers` (this package)
- `pydantic-evals` (for evaluation examples)
- `rapidfuzz` (for fuzzy matching, installed automatically)

## Need Help?

1. **Start with `fuzzy_quick_start.py`** for fuzzy matching
2. **Check `basic_usage.py`** for History functionality
3. **See the main README** for full documentation
4. **Look at test files** for more detailed examples
