"""pydantic-ai-helpers: Boring, opinionated helpers for PydanticAI.

Unofficial helpers that are so dumb you didn't want to implement them. So I did.

This is NOT an official PydanticAI package - just a simple personal helper library.

Key Modules
-----------
- history: Fluent API for accessing conversation history
- evals: Utilities for building robust evaluators with pydantic-evals
"""

from pydantic_ai_helpers.history import History

__version__ = "0.0.2"
__all__ = ["History"]
