# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is pydantic-ai-helpers, an unofficial library providing "boring, opinionated helpers" for PydanticAI. The library focuses on two main areas:

1. **History API** (`src/pydantic_ai_helpers/history.py`) - Fluent, chainable API for accessing PydanticAI conversation history
2. **Evaluation Utilities** (`src/pydantic_ai_helpers/evals/`) - Reusable evaluators with fuzzy string matching for comparing AI outputs

## Development Commands

### Essential Commands
- `make test` - Run tests with coverage (requires 100% coverage)
- `make lint` - Run ruff linting checks
- `make type` - Run mypy type checking
- `make format` - Format code with ruff
- `make check` - Run all checks (lint, type, test)

### Development Setup
- `make install-dev` - Install in development mode with all dependencies
- `make pre-commit` - Run pre-commit hooks on all files

### Other Commands
- `pytest tests/test_history.py::test_specific_function` - Run specific test
- `pytest tests/evals/` - Run just the evals tests
- `make coverage` - Generate and open HTML coverage report
- `make clean` - Remove build artifacts and cache files

## Architecture

### Core Components

**History Module** (`history.py`):
- `History` class: Main wrapper that accepts RunResult, StreamedRunResult, or list[ModelMessage]
- `RoleView`: Filtered access to messages by role (user, ai, system)
- `ToolsView`: Access tool calls and returns with optional name filtering
- `ToolPartView`: Filtered view of tool calls/returns
- `MediaView`: Access media content (images, audio, documents, videos) from user messages

**Evals Module** (`evals/`):
- `compare.py`: Core comparison utilities (ScalarCompare, ListCompare, InclusionCompare)
- `evaluators.py`: Pre-built evaluators for common patterns (ScalarEquals, ListRecall, etc.)
- `normalize.py`: Text normalization and fuzzy matching with rapidfuzz
- `accessors.py`: Safe dotted-path field access utilities
- `registry.py`: Registration system for evaluator specs

### Key Design Patterns

1. **Fluent API**: Chain method calls like `hist.tools.calls(name="dice").last()`
2. **Type Safety**: Full type hints throughout, leveraging PydanticAI's message types
3. **Autocomplete-Friendly**: IDE suggestions guide usage without documentation
4. **Immutable**: History objects never modify source data
5. **Fuzzy by Default**: Evaluation utilities enable fuzzy string matching (0.85 threshold) by default

### Dependencies
- `pydantic-ai>=0.0.20` - Core PydanticAI integration
- `pydantic-evals>=0.7.2` - Evaluation framework integration
- `rapidfuzz>=3.13.0` - Fuzzy string matching algorithms

### Test Structure
- `tests/test_history.py` - History API tests
- `tests/evals/` - Evaluation utilities tests
- `tests/test_conversations/` - JSON test conversation fixtures
- 100% test coverage requirement enforced by pytest-cov

### Code Quality
- ruff for linting and formatting (88 character line length)
- mypy for strict type checking
- pre-commit hooks for code quality
- numpy-style docstrings (pydocstyle)

## Special Notes

- The library automatically converts tool call string args to dictionary args when they contain valid JSON
- Fuzzy matching uses `token_set_ratio` algorithm by default for best word-based matching
- All text normalization happens before fuzzy matching for optimal results
- Media content filtering supports `url_only` and `binary_only` parameters for type-specific access
