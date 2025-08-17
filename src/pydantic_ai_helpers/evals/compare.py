# ruff: noqa: PLR0912
"""Comparison utilities for scalar, list, and inclusion evaluations.

This module provides flexible comparators that handle type coercion,
normalization, and various matching strategies with detailed reasoning
for evaluation results.

Examples
--------
    >>> comp = ScalarCompare(coerce_to="float", abs_tol=0.1)
    >>> value, reason = comp("3.14", 3.1)
    >>> print(value, reason)  # 1.0, "numbers match"

    >>> comp = ListCompare(mode="recall")
    >>> value, reason = comp(["a", "b"], ["a", "b", "c"])
    >>> print(value, reason)  # 0.6667, "recall: hits=2, denom=3, score=0.6667"
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any, Literal

from .normalize import (
    CompareOptions,
    FuzzyOptions,
    NormalizeOptions,
    NormalizeOpts,
    fuzzy_match_with_options,
    maybe_text_normalize_with_options,
)


def _repr_trunc(value: Any, max_len: int = 120) -> str:
    """Truncate repr to avoid overly long error messages.

    Parameters
    ----------
    value : Any
        Value to create a truncated repr for.
    max_len : int, default 120
        Maximum length of the repr string.

    Returns
    -------
    str
        Truncated representation of the value.
    """
    r = repr(value)
    return r if len(r) <= max_len else r[: max_len - 1] + "…"


# ---------- Type coercion utilities ----------


class CoercionError(ValueError):
    """Raised when type coercion fails."""


def _coerce_bool(x: Any) -> bool:
    """Coerce various types to boolean.

    Parameters
    ----------
    x : Any
        Value to coerce to boolean.

    Returns
    -------
    bool
        Coerced boolean value.

    Raises
    ------
    CoercionError
        If coercion is not possible.
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, int | float):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
    raise CoercionError(f"cannot coerce {x!r} to bool")


def _coerce(
    x: Any,
    to: Literal["str", "int", "float", "bool", "enum"] | type[object],
    *,
    enum_values: Sequence[str] | None = None,
) -> Any:
    """Coerce a value to a target type.

    Parameters
    ----------
    x : Any
        Value to coerce.
    to : str | Type[object]
        Target type. Can be "str", "int", "float", "bool", "enum", or an Enum class.
    enum_values : Sequence[str] | None, default None
        Valid enum values for enum coercion.

    Returns
    -------
    Any
        Coerced value.

    Raises
    ------
    CoercionError
        If coercion fails or target type is unsupported.
    """
    if isinstance(to, str):
        kind = to
    elif isinstance(to, type):
        # Allow Enum class passed in; compare by name
        if hasattr(to, "__members__"):  # Enum
            enum_values = list(to.__members__.keys())
            kind = "enum"
        else:
            kind = to.__name__.lower()
    else:
        raise CoercionError(f"unknown coercion target {to!r}")

    if kind == "str":
        return str(x)
    if kind == "int":
        return int(x)
    if kind == "float":
        return float(x)
    if kind == "bool":
        return _coerce_bool(x)
    if kind == "enum":
        if enum_values is None:
            raise CoercionError("enum_values required for enum comparison")
        s = str(x)
        # Compare by string identity; case-insensitive
        s_norm = s.strip().lower()
        for name in enum_values:
            if s_norm == name.lower():
                return name  # normalized enum member name
        raise CoercionError(f"{x!r} not in enum {sorted(enum_values)!r}")
    raise CoercionError(f"unsupported coercion target {to!r}")


# ---------- Comparator classes ----------


@dataclass(frozen=True)
class ScalarCompare:
    """Compare two scalar values with type coercion, tolerance, and fuzzy matching.

    Supports various types including strings, numbers, booleans, and enums.
    Provides detailed reasoning for evaluation results. Uses new structured
    options system with support for fuzzy string matching.

    Parameters
    ----------
    options : CompareOptions | None, default None
        Structured comparison options. If None, defaults are used.
    coerce_to : str | type | None, default None
        Target type for coercion (deprecated, use options.coerce_to).
    abs_tol : float | None, default None
        Absolute tolerance for numeric comparisons (deprecated, use options.abs_tol).
    rel_tol : float | None, default None
        Relative tolerance for numeric comparisons (deprecated, use options.rel_tol).
    normalize_opts : Mapping[str, Any] | None, default None
        String normalization options (deprecated, use options.normalize).
    enum_values : Sequence[str] | None, default None
        Valid enum values if coerce_to is "enum" (deprecated, use options.enum_values).

    Examples
    --------
    >>> # Using new structured options (preferred)
    >>> opts = CompareOptions(
    ...     normalize=NormalizeOptions(lowercase=True),
    ...     fuzzy=FuzzyOptions(enabled=True, threshold=0.85)
    ... )
    >>> comp = ScalarCompare(options=opts)
    >>> value, reason = comp("Hello World", "hello world")
    >>> print(value, reason)  # 1.0, "fuzzy match (score=1.0)"

    >>> # Numeric comparison with tolerance
    >>> opts = CompareOptions(coerce_to="float", abs_tol=0.1)
    >>> comp = ScalarCompare(options=opts)
    >>> value, reason = comp("3.14", 3.1)
    >>> print(value, reason)  # 1.0, "numbers match"

    >>> # Fuzzy string matching
    >>> comp = ScalarCompare()  # Uses defaults (fuzzy enabled)
    >>> value, reason = comp("apple", "aple")  # Typo
    >>> # Returns fuzzy score above threshold
    """

    options: CompareOptions | None = None
    # Deprecated parameters for backward compatibility
    coerce_to: Literal["str", "int", "float", "bool", "enum"] | type | None = None
    abs_tol: float | None = None
    rel_tol: float | None = None
    normalize_opts: NormalizeOpts | None = None
    enum_values: Sequence[str] | None = None

    def _get_effective_options(self) -> CompareOptions:
        """Get effective options, merging structured and deprecated parameters."""
        if self.options is not None:
            return self.options

        # Check if any deprecated parameters are being used
        using_deprecated_params = any(
            [
                self.coerce_to is not None,
                self.abs_tol is not None,
                self.rel_tol is not None,
                self.normalize_opts is not None,
                self.enum_values is not None,
            ]
        )

        # Build options from deprecated parameters
        normalize = None
        if self.normalize_opts:
            normalize = NormalizeOptions(
                lowercase=self.normalize_opts.get("lowercase", True),
                strip=self.normalize_opts.get("strip", True),
                collapse_spaces=self.normalize_opts.get("collapse_spaces", True),
                alphanum=self.normalize_opts.get("alphanum", False),
            )

        return CompareOptions(
            normalize=normalize,
            # Disable fuzzy for backwards compatibility when using deprecated params,
            # enable by default for new API
            fuzzy=FuzzyOptions(enabled=not using_deprecated_params),
            coerce_to=self.coerce_to,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
            enum_values=list(self.enum_values) if self.enum_values else None,
        )

    def __call__(self, left: Any, right: Any) -> tuple[Any, str]:  # noqa: PLR0911
        """Compare two values with configured options.

        Parameters
        ----------
        left : Any
            First value to compare.
        right : Any
            Second value to compare.

        Returns
        -------
        tuple[Any, str]
            Tuple of (score, reason) where score is 0.0-1.0 and reason
            is a human-readable explanation. For fuzzy matches, returns
            actual similarity score rather than binary 0/1.
        """
        opts = self._get_effective_options()

        # Use default options if not specified
        normalize_opts = opts.normalize or NormalizeOptions()
        fuzzy_opts = opts.fuzzy or FuzzyOptions()

        # Normalize text before coercion
        left_val = maybe_text_normalize_with_options(left, normalize_opts)
        right_val = maybe_text_normalize_with_options(right, normalize_opts)

        # Attempt coercion
        if opts.coerce_to is not None:
            try:
                left_val = _coerce(
                    left_val, opts.coerce_to, enum_values=opts.enum_values
                )
            except Exception as e:
                return 0.0, f"left coercion failed: {e}"
            try:
                right_val = _coerce(
                    right_val, opts.coerce_to, enum_values=opts.enum_values
                )
            except Exception as e:
                return 0.0, f"right coercion failed: {e}"

        # Numbers with tolerance
        if isinstance(left_val, int | float) and isinstance(right_val, int | float):
            if not (isfinite(float(left_val)) and isfinite(float(right_val))):
                return (
                    0.0,
                    f"non-finite number(s): left={left_val!r}, right={right_val!r}",
                )
            abs_tol = opts.abs_tol if opts.abs_tol is not None else 0.0
            rel_tol = opts.rel_tol if opts.rel_tol is not None else 0.0
            ok = isclose(
                float(left_val), float(right_val), rel_tol=rel_tol, abs_tol=abs_tol
            )
            value = 1.0 if ok else 0.0
            reason = (
                "numbers match"
                if ok
                else f"numbers differ: {left_val!r} vs {right_val!r} "
                f"(abs_tol={abs_tol}, rel_tol={rel_tol})"
            )
            return value, reason

        # String comparison with optional fuzzy matching
        if isinstance(left_val, str) and isinstance(right_val, str):
            if fuzzy_opts.enabled:
                # Use fuzzy matching - normalization already applied above
                score, is_match = fuzzy_match_with_options(
                    str(left), str(right), normalize_opts, fuzzy_opts
                )
                if is_match:
                    reason = (
                        f"fuzzy match (score={score:.3f}, "
                        f"threshold={fuzzy_opts.threshold})"
                    )
                else:
                    reason = (
                        f"fuzzy no match (score={score:.3f}, "
                        f"threshold={fuzzy_opts.threshold})"
                    )
                return score, reason
            else:
                # Exact match after normalization (already done above)
                ok = left_val == right_val
                return (
                    1.0 if ok else 0.0,
                    "values equal"
                    if ok
                    else f"values differ: {_repr_trunc(left_val)} vs "
                    f"{_repr_trunc(right_val)}",
                )

        # Everything else: exact equality after normalization/coercion
        ok = left_val == right_val
        return (
            1.0 if ok else 0.0,
            "values equal"
            if ok
            else f"values differ: {_repr_trunc(left_val)} vs {_repr_trunc(right_val)}",
        )


@dataclass(frozen=True)
class ListCompare:
    """Compare two sequences using various matching strategies with fuzzy support.

    Supports equality (with order sensitivity), recall, and precision metrics.
    Can handle both set and multiset semantics with normalization and fuzzy
    matching. When fuzzy matching is enabled, actual similarity scores are
    used instead of binary matches.

    Parameters
    ----------
    options : CompareOptions | None, default None
        Structured comparison options. If None, defaults are used.
    mode : str, default "equality"
        Comparison mode: "equality", "recall", or "precision".
    order_sensitive : bool, default False
        Whether order matters for equality comparisons.
    multiset : bool, default False
        If True, count duplicates; if False, use set semantics.
    normalize_opts : Mapping[str, Any] | None, default None
        Options for element normalization (deprecated, use options.normalize).
    element_coerce_to : str | type | None, default None
        Type to coerce elements to before comparison
        (deprecated, use options.coerce_to).

    Examples
    --------
    >>> # Using new structured options with fuzzy matching (preferred)
    >>> opts = CompareOptions(
    ...     normalize=NormalizeOptions(),
    ...     fuzzy=FuzzyOptions(enabled=True, threshold=0.85)
    ... )
    >>> comp = ListCompare(options=opts, mode="recall")
    >>> value, reason = comp(["apple", "banna"], ["apple", "banana", "cherry"])
    >>> # Returns fuzzy scores: "banna" matches "banana" with fuzzy score

    >>> # Exact matching (fuzzy disabled)
    >>> opts = CompareOptions(fuzzy=FuzzyOptions(enabled=False))
    >>> comp = ListCompare(options=opts, mode="equality", order_sensitive=True)
    >>> value, reason = comp(["a", "b"], ["b", "a"])
    >>> print(value, reason)  # 0.0, "lists differ"

    >>> # Default behavior (fuzzy enabled with normalization)
    >>> comp = ListCompare(mode="precision")  # Uses defaults
    >>> value, reason = comp(["Apple", "BANANA"], ["apple", "banana"])
    >>> # Returns 1.0 due to normalization + fuzzy matching
    """

    options: CompareOptions | None = None
    mode: Literal["equality", "recall", "precision"] = "equality"
    order_sensitive: bool = False
    multiset: bool = False
    # Deprecated parameters for backward compatibility
    normalize_opts: NormalizeOpts | None = None
    element_coerce_to: Literal["str", "int", "float", "bool", "enum"] | type | None = (
        None
    )

    def _get_effective_options(self) -> CompareOptions:
        """Get effective options, merging structured and deprecated parameters."""
        if self.options is not None:
            return self.options

        # Check if any deprecated parameters are being used
        using_deprecated_params = any(
            [
                self.normalize_opts is not None,
                self.element_coerce_to is not None,
            ]
        )

        # Build options from deprecated parameters
        normalize = None
        if self.normalize_opts:
            normalize = NormalizeOptions(
                lowercase=self.normalize_opts.get("lowercase", True),
                strip=self.normalize_opts.get("strip", True),
                collapse_spaces=self.normalize_opts.get("collapse_spaces", True),
                alphanum=self.normalize_opts.get("alphanum", False),
            )

        return CompareOptions(
            normalize=normalize,
            # Disable fuzzy for backwards compatibility when using deprecated params,
            # enable by default for new API
            fuzzy=FuzzyOptions(enabled=not using_deprecated_params),
            coerce_to=self.element_coerce_to,
        )

    def _prep(self, xs: Iterable[Any], opts: CompareOptions) -> list[Any]:
        """Prepare elements for comparison by normalizing and coercing.

        Parameters
        ----------
        xs : Iterable[Any]
            Elements to prepare.
        opts : CompareOptions
            Options for normalization and coercion.

        Returns
        -------
        list[Any]
            Prepared elements.
        """
        normalize_opts = opts.normalize or NormalizeOptions()
        xs = [maybe_text_normalize_with_options(x, normalize_opts) for x in xs]

        if opts.coerce_to is not None:
            out = []
            for x in xs:
                try:
                    out.append(_coerce(x, opts.coerce_to, enum_values=opts.enum_values))
                except Exception:
                    # Keep uncoercible as-is to make mismatch visible in reason
                    out.append(x)
            return out
        return xs

    def _fuzzy_find_best_match(
        self, item: Any, candidates: list[Any], opts: CompareOptions
    ) -> tuple[float, int | None]:
        """Find best fuzzy match for item in candidates.

        Parameters
        ----------
        item : Any
            Item to find match for.
        candidates : list[Any]
            List of candidate items to match against.
        opts : CompareOptions
            Options including fuzzy settings.

        Returns
        -------
        tuple[float, int | None]
            Tuple of (best_score, best_index). best_index is None if no
            match above threshold is found.
        """
        normalize_opts = opts.normalize or NormalizeOptions()
        fuzzy_opts = opts.fuzzy or FuzzyOptions()

        if not fuzzy_opts.enabled or not isinstance(item, str):
            # Exact matching fallback
            for i, candidate in enumerate(candidates):
                if item == candidate:
                    return 1.0, i
            return 0.0, None

        best_score = 0.0
        best_idx = None

        for i, candidate in enumerate(candidates):
            if isinstance(candidate, str):
                score, is_match = fuzzy_match_with_options(
                    str(item), str(candidate), normalize_opts, fuzzy_opts
                )
                if score > best_score:
                    best_score = score
                    best_idx = i if is_match else None
            elif item == candidate:
                # Exact match for non-strings
                return 1.0, i

        return best_score, best_idx

    def _to_bag(self, xs: Iterable[Any]) -> Mapping[Any, int]:
        """Convert sequence to multiset (bag) representation.

        Parameters
        ----------
        xs : Iterable[Any]
            Elements to count.

        Returns
        -------
        Mapping[Any, int]
            Mapping from element to count.
        """
        return Counter(xs)

    def __call__(self, left: Any, right: Any) -> tuple[Any, str]:  # noqa: PLR0911,PLR0915
        """Compare two sequences with configured options.

        Parameters
        ----------
        left : Any
            First sequence to compare.
        right : Any
            Second sequence to compare.

        Returns
        -------
        tuple[Any, str]
            Tuple of (score, reason) where score depends on mode and reason
            explains the comparison result. With fuzzy matching, returns
            actual similarity scores rather than binary matches.
        """
        if not isinstance(left, Iterable) or isinstance(left, str | bytes):
            return 0.0, f"left is not a sequence: {type(left).__name__}"
        if not isinstance(right, Iterable) or isinstance(right, str | bytes):
            return 0.0, f"right is not a sequence: {type(right).__name__}"

        opts = self._get_effective_options()
        fuzzy_opts = opts.fuzzy or FuzzyOptions()

        L = self._prep(list(left), opts)
        R = self._prep(list(right), opts)

        if self.mode == "equality":
            if fuzzy_opts.enabled and any(isinstance(x, str) for x in L + R):
                # Fuzzy equality: compare elements and average scores
                if len(L) != len(R):
                    return 0.0, f"lists have different lengths: {len(L)} vs {len(R)}"

                if self.order_sensitive:
                    # Order-sensitive fuzzy comparison
                    total_score = 0.0
                    for l_item, r_item in zip(L, R, strict=False):
                        if isinstance(l_item, str) and isinstance(r_item, str):
                            score, _ = fuzzy_match_with_options(
                                l_item, r_item, opts.normalize, fuzzy_opts
                            )
                        else:
                            score = 1.0 if l_item == r_item else 0.0
                        total_score += score

                    avg_score = total_score / len(L) if L else 1.0
                    # is_match = avg_score >= fuzzy_opts.threshold
                    return (
                        avg_score,
                        f"fuzzy equality: avg_score={avg_score:.3f}, "
                        f"threshold={fuzzy_opts.threshold}",
                    )
                else:
                    # Order-insensitive fuzzy comparison
                    used_indices = set()
                    total_score = 0.0

                    for l_item in L:
                        best_score, best_idx = self._fuzzy_find_best_match(
                            l_item,
                            [r for i, r in enumerate(R) if i not in used_indices],
                            opts,
                        )
                        if best_idx is not None:
                            # Adjust index to account for used items
                            available_indices = [
                                i for i in range(len(R)) if i not in used_indices
                            ]
                            actual_idx = available_indices[best_idx]
                            used_indices.add(actual_idx)
                        total_score += best_score

                    avg_score = total_score / max(len(L), len(R)) if (L or R) else 1.0
                    return (
                        avg_score,
                        f"fuzzy equality (unordered): avg_score={avg_score:.3f}",
                    )
            else:
                # Exact equality
                if self.multiset:
                    ok = self._to_bag(L) == self._to_bag(R)
                else:
                    ok = (L == R) if self.order_sensitive else (set(L) == set(R))
                return (
                    1.0 if ok else 0.0,
                    "lists equal"
                    if ok
                    else f"lists differ: left={_repr_trunc(L)}, right={_repr_trunc(R)}",
                )

        # Precision/recall with fuzzy matching
        if fuzzy_opts.enabled and any(isinstance(x, str) for x in L + R):
            # Fuzzy precision/recall: sum of fuzzy scores
            L_copy = L.copy()
            R_copy = R.copy()
            used_R_indices = set()
            total_fuzzy_score = 0.0
            matches = 0

            if self.mode == "recall":
                # For each item in R (expected), find best match in L (output)
                for r_item in R_copy:
                    available_L = [
                        item for i, item in enumerate(L_copy) if i not in used_R_indices
                    ]
                    best_score, best_idx = self._fuzzy_find_best_match(
                        r_item, available_L, opts
                    )
                    if best_score >= fuzzy_opts.threshold:
                        total_fuzzy_score += best_score
                        matches += 1
                        if best_idx is not None:
                            # Find actual index in original L
                            available_indices = [
                                i for i in range(len(L_copy)) if i not in used_R_indices
                            ]
                            if best_idx < len(available_indices):
                                used_R_indices.add(available_indices[best_idx])

                denom = len(R_copy)
            else:  # precision
                # For each item in L (output), find best match in R (expected)
                for l_item in L_copy:
                    available_R = [
                        r for i, r in enumerate(R_copy) if i not in used_R_indices
                    ]
                    best_score, best_idx = self._fuzzy_find_best_match(
                        l_item, available_R, opts
                    )
                    if best_score >= fuzzy_opts.threshold:
                        total_fuzzy_score += best_score
                        matches += 1
                        if best_idx is not None:
                            # Find actual index in original R
                            available_indices = [
                                i for i in range(len(R_copy)) if i not in used_R_indices
                            ]
                            if best_idx < len(available_indices):
                                used_R_indices.add(available_indices[best_idx])

                denom = len(L_copy)

            if denom == 0:
                return 1.0, f"fuzzy {self.mode}: denominator=0 (no requirements)"

            # Use sum of fuzzy scores / denominator (stricter than binary)
            score = total_fuzzy_score / denom
            return (
                score,
                f"fuzzy {self.mode}: fuzzy_score_sum={total_fuzzy_score:.3f}, "
                f"denom={denom}, score={score:.4f}",
            )
        else:
            # Exact precision/recall (original logic)
            if self.multiset:
                # Multiset intersection size
                cL, cR = Counter(L), Counter(R)
                inter = sum((cL & cR).values())
                denom = sum(cR.values()) if self.mode == "recall" else sum(cL.values())
            else:
                sL, sR = set(L), set(R)
                inter = len(sL & sR)
                denom = len(sR) if self.mode == "recall" else len(sL)

            if denom == 0:
                # Convention: no requirements -> perfect score
                return 1.0, f"{self.mode}: denominator=0 (no requirements)"

            score = inter / denom
            return score, f"{self.mode}: hits={inter}, denom={denom}, score={score:.4f}"


@dataclass(frozen=True)
class InclusionCompare:
    """Check if a single value is included in a sequence with fuzzy matching.

    Useful for checking if an output value is in a list of acceptable values,
    with support for normalization, type coercion, and fuzzy string matching.
    Returns the best fuzzy match score found.

    Parameters
    ----------
    options : CompareOptions | None, default None
        Structured comparison options. If None, defaults are used.
    normalize_opts : Mapping[str, Any] | None, default None
        Options for normalization (deprecated, use options.normalize).
    element_coerce_to : str | type | None, default None
        Type to coerce values to before comparison (deprecated, use options.coerce_to).

    Examples
    --------
    >>> # Using new structured options with fuzzy matching (preferred)
    >>> opts = CompareOptions(
    ...     normalize=NormalizeOptions(),
    ...     fuzzy=FuzzyOptions(enabled=True, threshold=0.85)
    ... )
    >>> comp = InclusionCompare(options=opts)
    >>> value, reason = comp("Cola", ["coke", "cola", "pepsi"])
    >>> print(value, reason)  # 1.0, "fuzzy match: 'cola' best_score=1.0"

    >>> # Fuzzy matching with typos
    >>> comp = InclusionCompare()  # Uses defaults (fuzzy enabled)
    >>> value, reason = comp("aple", ["apple", "banana", "cherry"])
    >>> # Returns fuzzy score for best match with "apple"

    >>> # Exact matching (fuzzy disabled)
    >>> opts = CompareOptions(fuzzy=FuzzyOptions(enabled=False))
    >>> comp = InclusionCompare(options=opts)
    >>> value, reason = comp("Apple", ["apple", "banana"])
    >>> # Returns 1.0 due to normalization making them equal
    """

    options: CompareOptions | None = None
    # Deprecated parameters for backward compatibility
    normalize_opts: NormalizeOpts | None = None
    element_coerce_to: Literal["str", "int", "float", "bool", "enum"] | type | None = (
        None
    )

    def _get_effective_options(self) -> CompareOptions:
        """Get effective options, merging structured and deprecated parameters."""
        if self.options is not None:
            return self.options

        # Check if any deprecated parameters are being used
        using_deprecated_params = any(
            [
                self.normalize_opts is not None,
                self.element_coerce_to is not None,
            ]
        )

        # Build options from deprecated parameters
        normalize = None
        if self.normalize_opts:
            normalize = NormalizeOptions(
                lowercase=self.normalize_opts.get("lowercase", True),
                strip=self.normalize_opts.get("strip", True),
                collapse_spaces=self.normalize_opts.get("collapse_spaces", True),
                alphanum=self.normalize_opts.get("alphanum", False),
            )

        return CompareOptions(
            normalize=normalize,
            # Disable fuzzy for backwards compatibility when using deprecated params,
            # enable by default for new API
            fuzzy=FuzzyOptions(enabled=not using_deprecated_params),
            coerce_to=self.element_coerce_to,
        )

    def __call__(self, left: Any, right: Any) -> tuple[Any, str]:
        """Check if left value is included in right sequence.

        Parameters
        ----------
        left : Any
            Value to check for inclusion.
        right : Any
            Sequence to check inclusion in.

        Returns
        -------
        tuple[Any, str]
            Tuple of (score, reason) where score is the best fuzzy match score
            (0.0-1.0) and reason explains the result. For exact matches or when
            fuzzy is disabled, returns 1.0 or 0.0.
        """
        opts = self._get_effective_options()
        normalize_opts = opts.normalize or NormalizeOptions()
        fuzzy_opts = opts.fuzzy or FuzzyOptions()

        # Normalize and coerce the left value
        left_val = maybe_text_normalize_with_options(left, normalize_opts)
        try:
            if opts.coerce_to is not None:
                left_val = _coerce(
                    left_val, opts.coerce_to, enum_values=opts.enum_values
                )
        except Exception as e:
            return 0.0, f"left coercion failed: {e}"

        if not isinstance(right, Iterable) or isinstance(right, str | bytes):
            return 0.0, f"right is not a sequence: {type(right).__name__}"

        # Prepare elements from the right sequence
        elems = []
        for x in right:
            x_norm = maybe_text_normalize_with_options(x, normalize_opts)
            try:
                if opts.coerce_to is not None:
                    x_norm = _coerce(
                        x_norm, opts.coerce_to, enum_values=opts.enum_values
                    )
            except Exception:
                pass
            elems.append(x_norm)

        # Check for inclusion with fuzzy matching
        if fuzzy_opts.enabled and isinstance(left_val, str):
            best_score = 0.0
            best_match = None

            for elem in elems:
                if isinstance(elem, str):
                    score, is_match = fuzzy_match_with_options(
                        str(left), str(elem), normalize_opts, fuzzy_opts
                    )
                    if score > best_score:
                        best_score = score
                        if is_match:
                            best_match = elem
                elif elem == left_val:
                    # Exact match for non-strings
                    return (
                        1.0,
                        f"{_repr_trunc(left_val)} in {_repr_trunc(elems)} "
                        "(exact match)",
                    )

            is_match = best_score >= fuzzy_opts.threshold
            if is_match:
                return (
                    best_score,
                    f"fuzzy match: {_repr_trunc(left_val)} -> "
                    f"{_repr_trunc(best_match)} (score={best_score:.3f})",
                )
            else:
                return (
                    best_score,
                    f"no fuzzy match: {_repr_trunc(left_val)} "
                    f"best_score={best_score:.3f} < "
                    f"threshold={fuzzy_opts.threshold}",
                )
        else:
            # Exact matching
            ok = any(e == left_val for e in elems)
            return (
                1.0 if ok else 0.0,
                f"{_repr_trunc(left_val)} {'in' if ok else 'not in'} "
                f"{_repr_trunc(elems)}",
            )
