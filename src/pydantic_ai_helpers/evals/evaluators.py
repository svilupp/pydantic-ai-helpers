"""Reusable evaluator implementations for pydantic-evals.

This module provides a collection of evaluators that use the comparison
utilities to perform field-to-field evaluations with detailed reasoning.
All evaluators return EvaluationReason objects for transparency.

Examples
--------
    >>> evaluator = ScalarEquals(
    ...     output_path="price",
    ...     expected_path="price",
    ...     coerce_to="float",
    ...     abs_tol=0.01
    ... )
    >>> # Use with pydantic-evals Dataset...

    >>> evaluator = ListRecall(
    ...     output_path="colors",
    ...     expected_path="expected_colors",
    ...     normalize_opts={"lowercase": True}
    ... )
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from .accessors import Accessor
from .compare import InclusionCompare, ListCompare, ScalarCompare
from .normalize import CompareOptions, FuzzyOptions, NormalizeOptions, NormalizeOpts


def _reason_prefix(name: str | None) -> str:
    """Create a prefix for evaluation reasons based on the evaluation name.

    Parameters
    ----------
    name : str | None
        Evaluation name to prefix with.

    Returns
    -------
    str
        Prefix string, either "[name] " or empty string.
    """
    return f"[{name}] " if name else ""


def _build_compare_options_from_flat_params(
    base_options: CompareOptions | None = None,
    *,
    fuzzy_enabled: bool | None = None,
    fuzzy_threshold: float | None = None,
    fuzzy_algorithm: str | None = None,
    normalize_lowercase: bool | None = None,
    normalize_strip: bool | None = None,
    normalize_collapse_spaces: bool | None = None,
    normalize_alphanum: bool | None = None,
    coerce_to: str | type | None = None,
    abs_tol: float | None = None,
    rel_tol: float | None = None,
    enum_values: list[str] | None = None,
    normalize_opts: NormalizeOpts | None = None,
    element_coerce_to: str | type | None = None,
) -> CompareOptions:
    """Build CompareOptions from flat parameters with proper override logic.

    Parameters.
    ----------
    base_options : CompareOptions | None
        Base options to start with. If None, defaults are used.
    **kwargs : various
        Flat parameters to override base options.

    Returns
    -------
    CompareOptions
        Combined options with flat parameter overrides applied.
    """
    # Start with base options or defaults
    base_opts = base_options if base_options is not None else CompareOptions()

    # Get current sub-options or defaults
    normalize_opts_obj = base_opts.normalize or NormalizeOptions()
    fuzzy_opts_obj = base_opts.fuzzy or FuzzyOptions()

    # Apply normalize option overrides
    if any(
        x is not None
        for x in [
            normalize_lowercase,
            normalize_strip,
            normalize_collapse_spaces,
            normalize_alphanum,
        ]
    ):
        normalize_opts_obj = NormalizeOptions(
            lowercase=(
                normalize_lowercase
                if normalize_lowercase is not None
                else normalize_opts_obj.lowercase
            ),
            strip=(
                normalize_strip
                if normalize_strip is not None
                else normalize_opts_obj.strip
            ),
            collapse_spaces=(
                normalize_collapse_spaces
                if normalize_collapse_spaces is not None
                else normalize_opts_obj.collapse_spaces
            ),
            alphanum=(
                normalize_alphanum
                if normalize_alphanum is not None
                else normalize_opts_obj.alphanum
            ),
        )

    # Apply fuzzy option overrides
    if any(x is not None for x in [fuzzy_enabled, fuzzy_threshold, fuzzy_algorithm]):
        fuzzy_opts_obj = FuzzyOptions(
            enabled=(
                fuzzy_enabled if fuzzy_enabled is not None else fuzzy_opts_obj.enabled
            ),
            threshold=(
                fuzzy_threshold
                if fuzzy_threshold is not None
                else fuzzy_opts_obj.threshold
            ),
            algorithm=(
                fuzzy_algorithm
                if fuzzy_algorithm is not None
                else fuzzy_opts_obj.algorithm
            ),
        )

    # Handle deprecated normalize_opts
    if normalize_opts is not None:
        normalize_opts_obj = NormalizeOptions(
            lowercase=normalize_opts.get("lowercase", normalize_opts_obj.lowercase),
            strip=normalize_opts.get("strip", normalize_opts_obj.strip),
            collapse_spaces=normalize_opts.get(
                "collapse_spaces", normalize_opts_obj.collapse_spaces
            ),
            alphanum=normalize_opts.get("alphanum", normalize_opts_obj.alphanum),
        )

    return CompareOptions(
        normalize=normalize_opts_obj,
        fuzzy=fuzzy_opts_obj,
        coerce_to=(
            coerce_to
            if coerce_to is not None
            else element_coerce_to
            if element_coerce_to is not None
            else base_opts.coerce_to
        ),
        abs_tol=abs_tol if abs_tol is not None else base_opts.abs_tol,
        rel_tol=rel_tol if rel_tol is not None else base_opts.rel_tol,
        enum_values=enum_values if enum_values is not None else base_opts.enum_values,
    )


# Callable comparator that returns (score, reason)
Comparator = Callable[[Any, Any], tuple[float, str]]


@dataclass(repr=False)
class CompareFields(Evaluator[object, object, object]):
    """Generic field-to-field comparison evaluator.

    Extracts values using dotted paths from ctx.output and ctx.expected_output,
    then compares them with the provided comparator. This is the foundational
    evaluator that other specialized evaluators build upon.

    Parameters
    ----------
    output_path : str | None, default None
        Dotted path to extract value from ctx.output. None means use root object.
    expected_path : str | None, default None
        Dotted path to extract value from ctx.expected_output.
        None means use root object.
    comparator : Any, default None
        Comparator function with signature (left, right) -> (value, reason).
    evaluation_name : str | None, default None
        Name for this evaluation, used in reports and reasoning.

    Returns
    -------
    EvaluationReason
        Result with numeric score in value and detailed reason string.

    Examples
    --------
    >>> from .compare import ScalarCompare
    >>> evaluator = CompareFields(
    ...     output_path="user.age",
    ...     expected_path="user.age",
    ...     comparator=ScalarCompare(coerce_to="int"),
    ...     evaluation_name="age_match"
    ... )
    """

    output_path: str | None = None
    expected_path: str | None = None
    comparator: Comparator | None = None
    evaluation_name: str | None = field(default=None)

    def evaluate(
        self, ctx: EvaluatorContext[object, object, object]
    ) -> EvaluationReason:
        """Evaluate by extracting and comparing field values.

        Parameters
        ----------
        ctx : EvaluatorContext
            Evaluation context containing inputs, output, and expected_output.

        Returns
        -------
        EvaluationReason
            Evaluation result with score and detailed reasoning.
        """
        # Resolve output
        ok_l, left, r_l = Accessor(self.output_path).get(ctx.output)
        if not ok_l:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name) + f"output path error: {r_l}"
                ),
            )

        # Resolve expected
        if ctx.expected_output is None:
            return EvaluationReason(
                value=1.0,
                reason=(
                    _reason_prefix(self.evaluation_name) + "no expected_output, passing"
                ),
            )

        ok_r, right, r_r = Accessor(self.expected_path).get(ctx.expected_output)
        if not ok_r:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name) + f"expected path error: {r_r}"
                ),
            )

        # Compare
        if self.comparator is None:
            return EvaluationReason(
                value=0.0,
                reason=_reason_prefix(self.evaluation_name) + "no comparator provided",
            )

        value, why = self.comparator(left, right)
        return EvaluationReason(
            value=value, reason=_reason_prefix(self.evaluation_name) + why
        )


# ---- Specialized evaluators ----


@dataclass(repr=False)
class ScalarEquals(CompareFields):
    """Evaluator for scalar equality with type coercion, tolerance, and fuzzy matching.

    Convenient wrapper around CompareFields with ScalarCompare for
    comparing individual values like numbers, strings, booleans, or enums.
    Supports fuzzy string matching with configurable thresholds.

    Parameters
    ----------
    output_path : str | None, default None
        Path to value in output.
    expected_path : str | None, default None
        Path to value in expected output.
    evaluation_name : str | None, default None
        Name for this evaluation.

    # Structured options (preferred)
    compare_options : CompareOptions | None, default None
        Complete comparison configuration including normalization, fuzzy matching,
        type coercion, and tolerances.

    # Flat options (convenience parameters)
    fuzzy_enabled : bool | None, default None
        Enable fuzzy string matching. Overrides compare_options.fuzzy.enabled.
    fuzzy_threshold : float | None, default None
        Fuzzy matching threshold (0.0-1.0). Overrides compare_options.fuzzy.threshold.
    fuzzy_algorithm : str | None, default None
        Fuzzy algorithm to use. Overrides compare_options.fuzzy.algorithm.
    normalize_lowercase : bool | None, default None
        Convert to lowercase. Overrides compare_options.normalize.lowercase.
    normalize_strip : bool | None, default None
        Strip whitespace. Overrides compare_options.normalize.strip.
    normalize_collapse_spaces : bool | None, default None
        Collapse multiple spaces. Overrides compare_options.normalize.collapse_spaces.
    normalize_alphanum : bool | None, default None
        Keep only alphanumeric. Overrides compare_options.normalize.alphanum.
    coerce_to : str | type | None, default None
        Target type for coercion. Overrides compare_options.coerce_to.
    abs_tol : float | None, default None
        Absolute tolerance for numeric comparisons. Overrides compare_options.abs_tol.
    rel_tol : float | None, default None
        Relative tolerance for numeric comparisons. Overrides compare_options.rel_tol.
    enum_values : Iterable[str] | None, default None
        Valid enum values if coerce_to is "enum". Overrides compare_options.enum_values.

    # Deprecated parameters
    normalize_opts : Mapping[str, Any] | None, default None
        String normalization options (deprecated, use normalize_* parameters).

    Examples
    --------
    >>> # Using flat options (preferred for simple cases)
    >>> evaluator = ScalarEquals(
    ...     output_path="user_name",
    ...     expected_path="expected_name",
    ...     fuzzy_enabled=True,
    ...     fuzzy_threshold=0.9,
    ...     normalize_lowercase=True
    ... )

    >>> # Using structured options (preferred for complex cases)
    >>> opts = CompareOptions(
    ...     normalize=NormalizeOptions(lowercase=True, strip=True),
    ...     fuzzy=FuzzyOptions(enabled=True, threshold=0.85),
    ...     coerce_to="float",
    ...     abs_tol=0.01
    ... )
    >>> evaluator = ScalarEquals(
    ...     output_path="price",
    ...     expected_path="price",
    ...     compare_options=opts,
    ...     evaluation_name="price_match"
    ... )

    >>> # Default behavior (fuzzy enabled, normalization enabled)
    >>> evaluator = ScalarEquals(
    ...     output_path="category",
    ...     expected_path="category"
    ... )
    >>> # Uses fuzzy matching with 0.85 threshold, normalization enabled
    """

    # Structured options
    compare_options: CompareOptions | None = None

    # Flat options (override structured options)
    fuzzy_enabled: bool | None = None
    fuzzy_threshold: float | None = None
    fuzzy_algorithm: str | None = None
    normalize_lowercase: bool | None = None
    normalize_strip: bool | None = None
    normalize_collapse_spaces: bool | None = None
    normalize_alphanum: bool | None = None
    coerce_to: str | type | None = None
    abs_tol: float | None = None
    rel_tol: float | None = None
    enum_values: Iterable[str] | None = None

    # Deprecated
    normalize_opts: NormalizeOpts | None = None

    def __post_init__(self) -> None:
        """Initialize the ScalarCompare comparator with configured options."""
        compare_opts = _build_compare_options_from_flat_params(
            base_options=self.compare_options,
            fuzzy_enabled=self.fuzzy_enabled,
            fuzzy_threshold=self.fuzzy_threshold,
            fuzzy_algorithm=self.fuzzy_algorithm,
            normalize_lowercase=self.normalize_lowercase,
            normalize_strip=self.normalize_strip,
            normalize_collapse_spaces=self.normalize_collapse_spaces,
            normalize_alphanum=self.normalize_alphanum,
            coerce_to=self.coerce_to,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
            enum_values=(
                list(self.enum_values) if self.enum_values is not None else None
            ),
            normalize_opts=self.normalize_opts,
        )
        self.comparator = ScalarCompare(options=compare_opts)


@dataclass(repr=False)
class ListEquality(CompareFields):
    """Evaluator for list equality with normalization, order sensitivity, and fuzzy.

    Convenient wrapper around CompareFields with ListCompare in equality mode
    for comparing sequences of values. Supports fuzzy string matching for
    better robustness against typos and variations.

    Parameters
    ----------
    output_path : str | None, default None
        Path to list in output.
    expected_path : str | None, default None
        Path to list in expected output.
    evaluation_name : str | None, default None
        Name for this evaluation.
    order_sensitive : bool, default False
        Whether order matters for equality.
    multiset : bool, default False
        Whether to count duplicates (multiset) or use set semantics.

    # Structured options (preferred)
    compare_options : CompareOptions | None, default None
        Complete comparison configuration.

    # Flat options (convenience parameters)
    fuzzy_enabled : bool | None, default None
        Enable fuzzy string matching.
    fuzzy_threshold : float | None, default None
        Fuzzy matching threshold (0.0-1.0).
    fuzzy_algorithm : str | None, default None
        Fuzzy algorithm to use.
    normalize_lowercase : bool | None, default None
        Convert to lowercase.
    normalize_strip : bool | None, default None
        Strip whitespace.
    normalize_collapse_spaces : bool | None, default None
        Collapse multiple spaces.
    normalize_alphanum : bool | None, default None
        Keep only alphanumeric.
    element_coerce_to : str | type | None, default None
        Type to coerce elements to.

    # Deprecated parameters
    normalize_opts : Mapping[str, Any] | None, default None
        Element normalization options (deprecated).

    Examples
    --------
    >>> # Using flat options (preferred for simple cases)
    >>> evaluator = ListEquality(
    ...     output_path="tags",
    ...     expected_path="tags",
    ...     fuzzy_enabled=True,
    ...     normalize_lowercase=True,
    ...     evaluation_name="tags_match"
    ... )

    >>> # Default behavior (fuzzy enabled, normalization enabled)
    >>> evaluator = ListEquality(
    ...     output_path="predicted_labels",
    ...     expected_path="expected_labels"
    ... )
    """

    order_sensitive: bool = False
    multiset: bool = False

    # Structured options
    compare_options: CompareOptions | None = None

    # Flat options
    fuzzy_enabled: bool | None = None
    fuzzy_threshold: float | None = None
    fuzzy_algorithm: str | None = None
    normalize_lowercase: bool | None = None
    normalize_strip: bool | None = None
    normalize_collapse_spaces: bool | None = None
    normalize_alphanum: bool | None = None
    element_coerce_to: str | type | None = None

    # Deprecated
    normalize_opts: NormalizeOpts | None = None

    def __post_init__(self) -> None:
        """Initialize the ListCompare comparator in equality mode."""
        compare_opts = _build_compare_options_from_flat_params(
            base_options=self.compare_options,
            fuzzy_enabled=self.fuzzy_enabled,
            fuzzy_threshold=self.fuzzy_threshold,
            fuzzy_algorithm=self.fuzzy_algorithm,
            normalize_lowercase=self.normalize_lowercase,
            normalize_strip=self.normalize_strip,
            normalize_collapse_spaces=self.normalize_collapse_spaces,
            normalize_alphanum=self.normalize_alphanum,
            element_coerce_to=self.element_coerce_to,
            normalize_opts=self.normalize_opts,
        )
        self.comparator = ListCompare(
            options=compare_opts,
            mode="equality",
            order_sensitive=self.order_sensitive,
            multiset=self.multiset,
        )


@dataclass(repr=False)
class ListRecall(CompareFields):
    """Evaluator for list recall metric with fuzzy matching support.

    Measures what fraction of expected items are present in the output.
    Useful for checking coverage of required elements. With fuzzy matching,
    uses actual similarity scores rather than binary matches.

    Parameters
    ----------
    output_path : str | None, default None
        Path to list in output.
    expected_path : str | None, default None
        Path to list in expected output.
    evaluation_name : str | None, default None
        Name for this evaluation.
    multiset : bool, default False
        Whether to count duplicates or use set semantics.

    # Structured options (preferred)
    compare_options : CompareOptions | None, default None
        Complete comparison configuration.

    # Flat options (convenience parameters)
    fuzzy_enabled : bool | None, default None
        Enable fuzzy string matching.
    fuzzy_threshold : float | None, default None
        Fuzzy matching threshold (0.0-1.0).
    fuzzy_algorithm : str | None, default None
        Fuzzy algorithm to use.
    normalize_lowercase : bool | None, default None
        Convert to lowercase.
    normalize_strip : bool | None, default None
        Strip whitespace.
    normalize_collapse_spaces : bool | None, default None
        Collapse multiple spaces.
    normalize_alphanum : bool | None, default None
        Keep only alphanumeric.
    element_coerce_to : str | type | None, default None
        Type to coerce elements to.

    # Deprecated parameters
    normalize_opts : Mapping[str, Any] | None, default None
        Element normalization options (deprecated).

    Examples
    --------
    >>> # Default behavior (fuzzy enabled, normalization enabled)
    >>> evaluator = ListRecall(
    ...     output_path="predicted_labels",
    ...     expected_path="required_labels",
    ...     evaluation_name="label_recall"
    ... )

    >>> # Custom fuzzy settings
    >>> evaluator = ListRecall(
    ...     output_path="predicted_tags",
    ...     expected_path="required_tags",
    ...     fuzzy_threshold=0.9,
    ...     normalize_alphanum=True
    ... )
    """

    multiset: bool = False

    # Structured options
    compare_options: CompareOptions | None = None

    # Flat options
    fuzzy_enabled: bool | None = None
    fuzzy_threshold: float | None = None
    fuzzy_algorithm: str | None = None
    normalize_lowercase: bool | None = None
    normalize_strip: bool | None = None
    normalize_collapse_spaces: bool | None = None
    normalize_alphanum: bool | None = None
    element_coerce_to: str | type | None = None

    # Deprecated
    normalize_opts: NormalizeOpts | None = None

    def __post_init__(self) -> None:
        """Initialize the ListCompare comparator in recall mode."""
        compare_opts = _build_compare_options_from_flat_params(
            base_options=self.compare_options,
            fuzzy_enabled=self.fuzzy_enabled,
            fuzzy_threshold=self.fuzzy_threshold,
            fuzzy_algorithm=self.fuzzy_algorithm,
            normalize_lowercase=self.normalize_lowercase,
            normalize_strip=self.normalize_strip,
            normalize_collapse_spaces=self.normalize_collapse_spaces,
            normalize_alphanum=self.normalize_alphanum,
            element_coerce_to=self.element_coerce_to,
            normalize_opts=self.normalize_opts,
        )
        self.comparator = ListCompare(
            options=compare_opts,
            mode="recall",
            multiset=self.multiset,
        )


@dataclass(repr=False)
class ListPrecision(CompareFields):
    """Evaluator for list precision metric with fuzzy matching support.

    Measures what fraction of output items are valid (present in expected).
    Useful for checking accuracy of predictions. With fuzzy matching,
    uses actual similarity scores rather than binary matches.

    Parameters
    ----------
    output_path : str | None, default None
        Path to list in output.
    expected_path : str | None, default None
        Path to list in expected output.
    evaluation_name : str | None, default None
        Name for this evaluation.
    multiset : bool, default False
        Whether to count duplicates or use set semantics.

    # Structured options (preferred)
    compare_options : CompareOptions | None, default None
        Complete comparison configuration.

    # Flat options (convenience parameters)
    fuzzy_enabled : bool | None, default None
        Enable fuzzy string matching.
    fuzzy_threshold : float | None, default None
        Fuzzy matching threshold (0.0-1.0).
    fuzzy_algorithm : str | None, default None
        Fuzzy algorithm to use.
    normalize_lowercase : bool | None, default None
        Convert to lowercase.
    normalize_strip : bool | None, default None
        Strip whitespace.
    normalize_collapse_spaces : bool | None, default None
        Collapse multiple spaces.
    normalize_alphanum : bool | None, default None
        Keep only alphanumeric.
    element_coerce_to : str | type | None, default None
        Type to coerce elements to.

    # Deprecated parameters
    normalize_opts : Mapping[str, Any] | None, default None
        Element normalization options (deprecated).

    Examples
    --------
    >>> # Default behavior (fuzzy enabled, normalization enabled)
    >>> evaluator = ListPrecision(
    ...     output_path="predicted_labels",
    ...     expected_path="valid_labels",
    ...     evaluation_name="label_precision"
    ... )

    >>> # Strict fuzzy matching
    >>> evaluator = ListPrecision(
    ...     output_path="predicted_categories",
    ...     expected_path="valid_categories",
    ...     fuzzy_threshold=0.95
    ... )
    """

    multiset: bool = False

    # Structured options
    compare_options: CompareOptions | None = None

    # Flat options
    fuzzy_enabled: bool | None = None
    fuzzy_threshold: float | None = None
    fuzzy_algorithm: str | None = None
    normalize_lowercase: bool | None = None
    normalize_strip: bool | None = None
    normalize_collapse_spaces: bool | None = None
    normalize_alphanum: bool | None = None
    element_coerce_to: str | type | None = None

    # Deprecated
    normalize_opts: NormalizeOpts | None = None

    def __post_init__(self) -> None:
        """Initialize the ListCompare comparator in precision mode."""
        compare_opts = _build_compare_options_from_flat_params(
            base_options=self.compare_options,
            fuzzy_enabled=self.fuzzy_enabled,
            fuzzy_threshold=self.fuzzy_threshold,
            fuzzy_algorithm=self.fuzzy_algorithm,
            normalize_lowercase=self.normalize_lowercase,
            normalize_strip=self.normalize_strip,
            normalize_collapse_spaces=self.normalize_collapse_spaces,
            normalize_alphanum=self.normalize_alphanum,
            element_coerce_to=self.element_coerce_to,
            normalize_opts=self.normalize_opts,
        )
        self.comparator = ListCompare(
            options=compare_opts,
            mode="precision",
            multiset=self.multiset,
        )


@dataclass(repr=False)
class ValueInExpectedList(CompareFields):
    """Evaluator for checking if output value is in expected list with fuzzy matching.

    Useful for validating that output is one of several acceptable values,
    like checking if a predicted category is in the list of valid categories.
    With fuzzy matching, returns the best similarity score found.

    Parameters
    ----------
    output_path : str | None, default None
        Path to value in output.
    expected_path : str | None, default None
        Path to list in expected output.
    evaluation_name : str | None, default None
        Name for this evaluation.

    # Structured options (preferred)
    compare_options : CompareOptions | None, default None
        Complete comparison configuration.

    # Flat options (convenience parameters)
    fuzzy_enabled : bool | None, default None
        Enable fuzzy string matching.
    fuzzy_threshold : float | None, default None
        Fuzzy matching threshold (0.0-1.0).
    fuzzy_algorithm : str | None, default None
        Fuzzy algorithm to use.
    normalize_lowercase : bool | None, default None
        Convert to lowercase.
    normalize_strip : bool | None, default None
        Strip whitespace.
    normalize_collapse_spaces : bool | None, default None
        Collapse multiple spaces.
    normalize_alphanum : bool | None, default None
        Keep only alphanumeric.
    element_coerce_to : str | type | None, default None
        Type to coerce value and elements to.

    # Deprecated parameters
    normalize_opts : Mapping[str, Any] | None, default None
        Normalization options (deprecated).

    Examples
    --------
    >>> # Default behavior (fuzzy enabled, normalization enabled)
    >>> evaluator = ValueInExpectedList(
    ...     output_path="predicted_category",
    ...     expected_path="valid_categories",
    ...     evaluation_name="category_valid"
    ... )

    >>> # Custom fuzzy settings for strict validation
    >>> evaluator = ValueInExpectedList(
    ...     output_path="user_input",
    ...     expected_path="allowed_values",
    ...     fuzzy_threshold=0.9,
    ...     normalize_alphanum=True
    ... )
    """

    # Structured options
    compare_options: CompareOptions | None = None

    # Flat options
    fuzzy_enabled: bool | None = None
    fuzzy_threshold: float | None = None
    fuzzy_algorithm: str | None = None
    normalize_lowercase: bool | None = None
    normalize_strip: bool | None = None
    normalize_collapse_spaces: bool | None = None
    normalize_alphanum: bool | None = None
    element_coerce_to: str | type | None = None

    # Deprecated
    normalize_opts: NormalizeOpts | None = None

    def __post_init__(self) -> None:
        """Initialize the InclusionCompare comparator."""
        compare_opts = _build_compare_options_from_flat_params(
            base_options=self.compare_options,
            fuzzy_enabled=self.fuzzy_enabled,
            fuzzy_threshold=self.fuzzy_threshold,
            fuzzy_algorithm=self.fuzzy_algorithm,
            normalize_lowercase=self.normalize_lowercase,
            normalize_strip=self.normalize_strip,
            normalize_collapse_spaces=self.normalize_collapse_spaces,
            normalize_alphanum=self.normalize_alphanum,
            element_coerce_to=self.element_coerce_to,
            normalize_opts=self.normalize_opts,
        )
        self.comparator = InclusionCompare(options=compare_opts)


# MultiCompare removed: keep API simple and focused.


@dataclass(repr=False)
class MaxCount(Evaluator[object, object, object]):
    """Evaluator for checking if a numeric value is below or equal to a maximum count.

    Extracts a value from the output, converts it to an integer, and checks
    if it's less than or equal to the specified maximum count. Returns 1.0
    if the condition is met, 0.0 otherwise.

    Parameters
    ----------
    output_path : str | None, default None
        Dotted path to extract value from ctx.output. None means use root object.
    count : int
        Maximum allowed count (inclusive).
    evaluation_name : str | None, default None
        Name for this evaluation, used in reports and reasoning.

    Returns
    -------
    EvaluationReason
        Result with 1.0 if value <= count, 0.0 otherwise, with detailed reason.

    Examples
    --------
    >>> evaluator = MaxCount(
    ...     output_path="item_count",
    ...     count=10,
    ...     evaluation_name="max_items"
    ... )
    """

    output_path: str | None = None
    count: int = 0
    evaluation_name: str | None = field(default=None)

    def evaluate(
        self, ctx: EvaluatorContext[object, object, object]
    ) -> EvaluationReason:
        """Evaluate by checking if extracted value is <= count.

        Parameters
        ----------
        ctx : EvaluatorContext
            Evaluation context containing inputs, output, and expected_output.

        Returns
        -------
        EvaluationReason
            Evaluation result with score and detailed reasoning.
        """
        # Extract value from output
        ok, value, reason = Accessor(self.output_path).get(ctx.output)
        if not ok:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"output path error: {reason}"
                ),
            )

        # Convert to int
        try:
            int_value = int(value)
        except (ValueError, TypeError) as e:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"cannot convert '{value}' to int: {e}"
                ),
            )

        # Check condition
        if int_value <= self.count:
            return EvaluationReason(
                value=1.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"value {int_value} is <= max count {self.count}"
                ),
            )
        else:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"value {int_value} exceeds max count {self.count}"
                ),
            )


@dataclass(repr=False)
class MinCount(Evaluator[object, object, object]):
    """Evaluator for checking if a numeric value is above or equal to a minimum count.

    Extracts a value from the output, converts it to an integer, and checks
    if it's greater than or equal to the specified minimum count. Returns 1.0
    if the condition is met, 0.0 otherwise.

    Parameters
    ----------
    output_path : str | None, default None
        Dotted path to extract value from ctx.output. None means use root object.
    count : int
        Minimum required count (inclusive).
    evaluation_name : str | None, default None
        Name for this evaluation, used in reports and reasoning.

    Returns
    -------
    EvaluationReason
        Result with 1.0 if value >= count, 0.0 otherwise, with detailed reason.

    Examples
    --------
    >>> evaluator = MinCount(
    ...     output_path="item_count",
    ...     count=5,
    ...     evaluation_name="min_items"
    ... )
    """

    output_path: str | None = None
    count: int = 0
    evaluation_name: str | None = field(default=None)

    def evaluate(
        self, ctx: EvaluatorContext[object, object, object]
    ) -> EvaluationReason:
        """Evaluate by checking if extracted value is >= count.

        Parameters
        ----------
        ctx : EvaluatorContext
            Evaluation context containing inputs, output, and expected_output.

        Returns
        -------
        EvaluationReason
            Evaluation result with score and detailed reasoning.
        """
        # Extract value from output
        ok, value, reason = Accessor(self.output_path).get(ctx.output)
        if not ok:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"output path error: {reason}"
                ),
            )

        # Convert to int
        try:
            int_value = int(value)
        except (ValueError, TypeError) as e:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"cannot convert '{value}' to int: {e}"
                ),
            )

        # Check condition
        if int_value >= self.count:
            return EvaluationReason(
                value=1.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"value {int_value} is >= min count {self.count}"
                ),
            )
        else:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"value {int_value} is below min count {self.count}"
                ),
            )


@dataclass(repr=False)
class MaxLength(Evaluator[object, object, object]):
    """Evaluator for checking if a string length is below or equal to a maximum length.

    Extracts a value from the output, converts it to a string, strips whitespace,
    and checks if its length is less than or equal to the specified maximum length.
    Returns 1.0 if the condition is met, 0.0 otherwise.

    Parameters
    ----------
    output_path : str | None, default None
        Dotted path to extract value from ctx.output. None means use root object.
    length : int
        Maximum allowed length (inclusive) after stripping.
    evaluation_name : str | None, default None
        Name for this evaluation, used in reports and reasoning.

    Returns
    -------
    EvaluationReason
        Result with 1.0 if len(value.strip()) <= length, 0.0 otherwise, with reason.

    Examples
    --------
    >>> evaluator = MaxLength(
    ...     output_path="description",
    ...     length=100,
    ...     evaluation_name="max_desc_length"
    ... )
    """

    output_path: str | None = None
    length: int = 0
    evaluation_name: str | None = field(default=None)

    def evaluate(
        self, ctx: EvaluatorContext[object, object, object]
    ) -> EvaluationReason:
        """Evaluate by checking if extracted string length is <= length.

        Parameters
        ----------
        ctx : EvaluatorContext
            Evaluation context containing inputs, output, and expected_output.

        Returns
        -------
        EvaluationReason
            Evaluation result with score and detailed reasoning.
        """
        # Extract value from output
        ok, value, reason = Accessor(self.output_path).get(ctx.output)
        if not ok:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"output path error: {reason}"
                ),
            )

        # Convert to string and strip
        try:
            str_value = str(value).strip()
        except Exception as e:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"cannot convert '{value}' to string: {e}"
                ),
            )

        actual_length = len(str_value)

        # Check condition
        if actual_length <= self.length:
            return EvaluationReason(
                value=1.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"string length {actual_length} is <= max length {self.length}"
                ),
            )
        else:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"string length {actual_length} exceeds max length {self.length}"
                ),
            )


@dataclass(repr=False)
class MinLength(Evaluator[object, object, object]):
    """Evaluator for checking if a string length is above or equal to a minimum length.

    Extracts a value from the output, converts it to a string, strips whitespace,
    and checks if its length is greater than or equal to the specified minimum length.
    Returns 1.0 if the condition is met, 0.0 otherwise.

    Parameters
    ----------
    output_path : str | None, default None
        Dotted path to extract value from ctx.output. None means use root object.
    length : int
        Minimum required length (inclusive) after stripping.
    evaluation_name : str | None, default None
        Name for this evaluation, used in reports and reasoning.

    Returns
    -------
    EvaluationReason
        Result with 1.0 if len(value.strip()) >= length, 0.0 otherwise, with reason.

    Examples
    --------
    >>> evaluator = MinLength(
    ...     output_path="description",
    ...     length=10,
    ...     evaluation_name="min_desc_length"
    ... )
    """

    output_path: str | None = None
    length: int = 0
    evaluation_name: str | None = field(default=None)

    def evaluate(
        self, ctx: EvaluatorContext[object, object, object]
    ) -> EvaluationReason:
        """Evaluate by checking if extracted string length is >= length.

        Parameters
        ----------
        ctx : EvaluatorContext
            Evaluation context containing inputs, output, and expected_output.

        Returns
        -------
        EvaluationReason
            Evaluation result with score and detailed reasoning.
        """
        # Extract value from output
        ok, value, reason = Accessor(self.output_path).get(ctx.output)
        if not ok:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"output path error: {reason}"
                ),
            )

        # Convert to string and strip
        try:
            str_value = str(value).strip()
        except Exception as e:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"cannot convert '{value}' to string: {e}"
                ),
            )

        actual_length = len(str_value)

        # Check condition
        if actual_length >= self.length:
            return EvaluationReason(
                value=1.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"string length {actual_length} is >= min length {self.length}"
                ),
            )
        else:
            return EvaluationReason(
                value=0.0,
                reason=(
                    _reason_prefix(self.evaluation_name)
                    + f"string length {actual_length} is below min length {self.length}"
                ),
            )
