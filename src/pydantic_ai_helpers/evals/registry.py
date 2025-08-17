"""Registry utilities for managing evaluators with datasets.

This module provides convenience functions for registering evaluators
with pydantic-evals Dataset objects, including support for building
evaluators from configuration specifications.

Examples
--------
    >>> from pydantic_evals import Dataset
    >>> from .evaluators import ScalarEquals, ListRecall
    >>>
    >>> dataset = Dataset(cases=[...])
    >>> register(dataset,
    ...     ScalarEquals(output_path="price", expected_path="price"),
    ...     ListRecall(output_path="tags", expected_path="tags")
    ... )

    >>> # Or from specifications:
    >>> specs = [
    ...     {"kind": "ScalarEquals", "output_path": "price", "expected_path": "price"},
    ...     {"kind": "ListRecall", "output_path": "tags", "expected_path": "tags"}
    ... ]
    >>> from_specs(dataset, specs)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def register(dataset: Any, *evaluators: Any) -> None:
    """Append evaluators to a pydantic-evals Dataset-like object.

    This is a simple convenience function that adds multiple evaluators
    to a dataset in a single call.

    Parameters
    ----------
    dataset : Any
        Dataset object with an add_evaluator method.
    *evaluators : Any
        Evaluator instances to add to the dataset.

    Examples
    --------
    >>> from pydantic_evals import Dataset
    >>> from .evaluators import ScalarEquals, ListRecall
    >>>
    >>> dataset = Dataset(cases=[...])
    >>> register(dataset,
    ...     ScalarEquals(output_path="price", expected_path="price"),
    ...     ListRecall(output_path="tags", expected_path="tags")
    ... )
    """
    for ev in evaluators:
        dataset.add_evaluator(ev)


def from_specs(dataset: Any, specs: Iterable[dict[str, Any]]) -> None:
    """Build evaluators from specifications and register them with a dataset.

    This allows creating evaluators from configuration data like JSON or YAML,
    making it easy to define evaluation suites declaratively.

    Parameters
    ----------
    dataset : Any
        Dataset object with an add_evaluator method.
    specs : Iterable[dict[str, Any]]
        Iterable of specification dictionaries. Each spec should have a "kind"
        key indicating the evaluator type, and other keys for constructor arguments.

    Supported Evaluator Types
    -------------------------
    - "CompareFields": Generic field comparison
    - "ScalarEquals": Scalar value equality
    - "ListEquality": List equality comparison
    - "ListRecall": List recall metric
    - "ListPrecision": List precision metric
    - "ValueInExpectedList": Value inclusion check

    Special Keys
    ------------
    - "kind": Required. Evaluator class name.
    - "name": Optional. Sets evaluation_name attribute.
    - "normalize": Optional. Dict with normalization options (lowercase, strip, etc).
    - "fuzzy": Optional. Dict with fuzzy matching options
      (enabled, threshold, algorithm).

    Examples
    --------
    >>> specs = [
    ...     {
    ...         "kind": "ScalarEquals",
    ...         "name": "price_match",
    ...         "output_path": "price",
    ...         "expected_path": "price",
    ...         "coerce_to": "float",
    ...         "abs_tol": 0.01
    ...     },
    ...     {
    ...         "kind": "ListRecall",
    ...         "name": "tag_recall",
    ...         "output_path": "tags",
    ...         "expected_path": "expected_tags",
    ...         "normalize": {"lowercase": True, "strip": True},
    ...         "fuzzy": {"enabled": True, "threshold": 0.9}
    ...     },
    ...     {
    ...         "kind": "ValueInExpectedList",
    ...         "name": "category_check",
    ...         "output_path": "category",
    ...         "expected_path": "valid_categories",
    ...         "fuzzy_enabled": True,
    ...         "normalize_lowercase": True
    ...     }
    ... ]
    >>> from_specs(dataset, specs)
    """
    # Import here to avoid circular imports
    from .evaluators import (
        CompareFields,
        ListEquality,
        ListPrecision,
        ListRecall,
        ScalarEquals,
        ValueInExpectedList,
    )

    kind_map = {
        "CompareFields": CompareFields,
        "ScalarEquals": ScalarEquals,
        "ListEquality": ListEquality,
        "ListRecall": ListRecall,
        "ListPrecision": ListPrecision,
        "ValueInExpectedList": ValueInExpectedList,
    }

    for s in specs:
        # Make a copy to avoid modifying the original
        spec = dict(s)

        # Extract special keys
        kind = spec.pop("kind")
        if kind not in kind_map:
            raise ValueError(
                f"Unknown evaluator kind: {kind}. Supported: {list(kind_map.keys())}"
            )

        cls = kind_map[kind]
        name = spec.pop("name", None)
        normalize = spec.pop("normalize", None)

        # Map "normalize" to flat normalize parameters for consistency
        if normalize is not None:
            # Convert normalize dict to flat parameters
            if isinstance(normalize, dict):
                for key, value in normalize.items():
                    if key in {"lowercase", "strip", "collapse_spaces", "alphanum"}:
                        spec[f"normalize_{key}"] = value
            else:
                # Fallback to old format for backward compatibility
                spec["normalize_opts"] = normalize

        # Map "fuzzy" to flat fuzzy parameters
        fuzzy = spec.pop("fuzzy", None)
        if fuzzy is not None and isinstance(fuzzy, dict):
            for key, value in fuzzy.items():
                if key in {"enabled", "threshold", "algorithm"}:
                    spec[f"fuzzy_{key}"] = value

        # Create the evaluator
        ev = cls(**spec)

        # Set evaluation name if provided
        if name is not None:
            object.__setattr__(ev, "evaluation_name", name)

        # Register with dataset
        dataset.add_evaluator(ev)
