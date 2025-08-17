"""Evaluation utilities for pydantic-evals with fuzzy matching support.

This module provides utilities for building robust, reusable evaluators
that compare fields between output and expected output with support for
type coercion, normalization, fuzzy string matching, and detailed reasoning.

Key Features
------------
- Safe dotted-path field access (e.g., "user.profile.age")
- Fuzzy string matching with configurable algorithms and thresholds
- Consistent text and sequence normalization (enabled by default)
- Flexible comparison strategies (scalar, list, inclusion)
- Pre-built evaluators for common patterns
- Detailed reasoning for evaluation results
- Fully typed options system with IDE support

Examples
--------
Compare scalar fields with fuzzy matching (default behavior):

    >>> evaluator = ScalarEquals(
    ...     output_path="user.name",
    ...     expected_path="user.name"
    ... )
    >>> # Uses fuzzy matching with 0.85 threshold by default

Check list recall with custom fuzzy settings:

    >>> evaluator = ListRecall(
    ...     output_path="predicted_tags",
    ...     expected_path="required_tags",
    ...     fuzzy_threshold=0.9,
    ...     normalize_lowercase=True
    ... )

Validate value is in acceptable list with fuzzy matching:

    >>> evaluator = ValueInExpectedList(
    ...     output_path="category",
    ...     expected_path="valid_categories",
    ...     fuzzy_enabled=True,
    ...     normalize_alphanum=True
    ... )

Using structured options for complex configuration:

    >>> from pydantic_ai_helpers.evals import (
    ...     CompareOptions, FuzzyOptions, NormalizeOptions
    ... )
    >>> opts = CompareOptions(
    ...     normalize=NormalizeOptions(lowercase=True, strip=True),
    ...     fuzzy=FuzzyOptions(
    ...         enabled=True, threshold=0.85, algorithm="token_set_ratio"
    ...     )
    ... )
    >>> evaluator = ScalarEquals(
    ...     output_path="product_name",
    ...     expected_path="product_name",
    ...     compare_options=opts
    ... )

Disable fuzzy matching for exact comparisons:

    >>> evaluator = ListEquality(
    ...     output_path="exact_list",
    ...     expected_path="exact_list",
    ...     fuzzy_enabled=False
    ... )

Aggregate multiple evaluations:

    >>> # Aggregation helpers intentionally not provided to keep API simple.
"""

from .accessors import Accessor, resolve_path
from .compare import InclusionCompare, ListCompare, ScalarCompare
from .evaluators import (
    CompareFields,
    ListEquality,
    ListPrecision,
    ListRecall,
    ScalarEquals,
    ValueInExpectedList,
)
from .normalize import (
    CompareOptions,
    FuzzyOptions,
    NormalizeOptions,
    fuzzy_match_score,
    fuzzy_match_with_options,
    text_normalize,
    text_normalize_with_options,
)
from .registry import from_specs, register

__all__ = [
    "Accessor",
    "CompareFields",
    "CompareOptions",
    "FuzzyOptions",
    "InclusionCompare",
    "ListCompare",
    "ListEquality",
    "ListPrecision",
    "ListRecall",
    "NormalizeOptions",
    "ScalarCompare",
    "ScalarEquals",
    "ValueInExpectedList",
    "from_specs",
    "fuzzy_match_score",
    "fuzzy_match_with_options",
    "register",
    "resolve_path",
    "text_normalize",
    "text_normalize_with_options",
]
