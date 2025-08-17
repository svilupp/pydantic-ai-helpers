"""Text and sequence normalization utilities for evaluations.

This module provides consistent normalization functions that can be applied
to both individual values and sequences of values, ensuring fair comparisons
between expected and actual outputs.

Examples
--------
    >>> text_normalize("  Hello World!  ", lowercase=True, strip=True)
    'hello world!'

    >>> text_normalize("Hello-World_123", alphanum=True)
    'HelloWorld123'

    >>> # Using structured options
    >>> opts = NormalizeOptions(lowercase=True, strip=True)
    >>> text_normalize_with_options("  Hello World!  ", opts)
    'hello world!'
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, TypeVar

# Import rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz
except ImportError as e:
    raise ImportError(
        "rapidfuzz is required for fuzzy matching. Install with: pip install rapidfuzz"
    ) from e

T = TypeVar("T")

_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")


class NormalizeOpts(TypedDict, total=False):
    """Options for text normalization (deprecated, use NormalizeOptions).

    - lowercase: Convert to lowercase.
    - strip: Trim leading and trailing whitespace.
    - alphanum: Remove non-alphanumeric characters.
    - collapse_spaces: Collapse multiple spaces to a single space.
    """

    lowercase: bool
    strip: bool
    alphanum: bool
    collapse_spaces: bool


@dataclass
class NormalizeOptions:
    """Typed options for text normalization.

    Defaults follow common evaluation patterns where text is typically
    normalized for fair comparison.

    Parameters
    ----------
    lowercase : bool, default True
        Convert text to lowercase for case-insensitive comparison.
    strip : bool, default True
        Remove leading and trailing whitespace.
    collapse_spaces : bool, default True
        Collapse multiple consecutive spaces to single spaces.
    alphanum : bool, default False
        Remove all non-alphanumeric characters.

    Examples
    --------
    >>> opts = NormalizeOptions()  # Uses defaults
    >>> text_normalize_with_options("  Hello World!  ", opts)
    'hello world!'

    >>> opts = NormalizeOptions(alphanum=True)
    >>> text_normalize_with_options("Hello-World_123!", opts)
    'helloworld123'
    """

    lowercase: bool = True
    strip: bool = True
    collapse_spaces: bool = True
    alphanum: bool = False


@dataclass
class FuzzyOptions:
    """Typed options for fuzzy string matching.

    Fuzzy matching is enabled by default with sensible defaults for
    most evaluation scenarios.

    Parameters
    ----------
    enabled : bool, default True
        Whether to enable fuzzy matching.
    threshold : float, default 0.85
        Minimum similarity score (0-1) to consider a match.
        Higher values require closer matches.
    algorithm : str, default "token_set_ratio"
        Fuzzy matching algorithm to use:
        - "ratio": Simple character-based similarity
        - "partial_ratio": Best matching substring
        - "token_sort_ratio": Token-based matching with sorting
        - "token_set_ratio": Token-based matching with set logic

    Examples
    --------
    >>> opts = FuzzyOptions()  # Default: enabled, 0.85 threshold, token_set_ratio
    >>> opts = FuzzyOptions(threshold=0.9, algorithm="ratio")  # Stricter matching
    >>> opts = FuzzyOptions(enabled=False)  # Disable fuzzy matching
    """

    enabled: bool = True
    threshold: float = 0.85
    algorithm: Literal[
        "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"
    ] = "token_set_ratio"


@dataclass
class CompareOptions:
    """Combined options for comparison operations.

    Centralizes all comparison-related configuration including normalization,
    fuzzy matching, type coercion, and tolerance settings.

    Parameters
    ----------
    normalize : NormalizeOptions | None, default None
        Text normalization options. If None, defaults are used.
    fuzzy : FuzzyOptions | None, default None
        Fuzzy matching options. If None, defaults are used.
    coerce_to : str | type | None, default None
        Target type for value coercion before comparison.
    abs_tol : float | None, default None
        Absolute tolerance for numeric comparisons.
    rel_tol : float | None, default None
        Relative tolerance for numeric comparisons.
    enum_values : list[str] | None, default None
        Valid enum values for enum coercion.

    Examples
    --------
    >>> # Use all defaults
    >>> opts = CompareOptions()

    >>> # Custom normalization only
    >>> opts = CompareOptions(
    ...     normalize=NormalizeOptions(alphanum=True)
    ... )

    >>> # Disable fuzzy matching
    >>> opts = CompareOptions(
    ...     fuzzy=FuzzyOptions(enabled=False)
    ... )

    >>> # Numeric comparison with tolerance
    >>> opts = CompareOptions(
    ...     coerce_to="float",
    ...     abs_tol=0.01
    ... )
    """

    normalize: NormalizeOptions | None = None
    fuzzy: FuzzyOptions | None = None
    coerce_to: str | type | None = None
    abs_tol: float | None = None
    rel_tol: float | None = None
    enum_values: list[str] | None = None


def text_normalize(
    s: str,
    *,
    lowercase: bool = False,
    strip: bool = False,
    alphanum: bool = False,
    collapse_spaces: bool = False,
) -> str:
    """Normalize a string using various transformation options.

    Parameters
    ----------
    s : str
        String to normalize.
    lowercase : bool, default False
        Convert to lowercase.
    strip : bool, default False
        Remove leading and trailing whitespace.
    alphanum : bool, default False
        Remove all non-alphanumeric characters.
    collapse_spaces : bool, default False
        Collapse multiple consecutive spaces into single spaces.

    Returns
    -------
    str
        Normalized string.

    Examples
    --------
    >>> text_normalize("  Hello World!  ", lowercase=True, strip=True)
    'hello world!'

    >>> text_normalize("Hello-World_123", alphanum=True)
    'HelloWorld123'

    >>> text_normalize("Multiple   Spaces", collapse_spaces=True)
    'Multiple Spaces'
    """
    if strip:
        s = s.strip()
    if lowercase:
        s = s.lower()
    if alphanum:
        s = _ALNUM_RE.sub("", s)
    if collapse_spaces:
        s = " ".join(s.split())
    return s


def text_normalize_with_options(s: str, options: NormalizeOptions) -> str:
    """Normalize a string using structured options.

    Parameters
    ----------
    s : str
        String to normalize.
    options : NormalizeOptions
        Normalization options to apply.

    Returns
    -------
    str
        Normalized string.

    Examples
    --------
    >>> opts = NormalizeOptions()  # Default normalization
    >>> text_normalize_with_options("  Hello World!  ", opts)
    'hello world!'

    >>> opts = NormalizeOptions(alphanum=True)
    >>> text_normalize_with_options("Hello-World_123!", opts)
    'helloworld123'
    """
    return text_normalize(
        s,
        lowercase=options.lowercase,
        strip=options.strip,
        alphanum=options.alphanum,
        collapse_spaces=options.collapse_spaces,
    )


def maybe_text_normalize(
    x: Any,
    **opts: Any,
) -> Any:
    """Apply text normalization only if the value is a string.

    This is useful for handling mixed-type sequences where you want
    to normalize strings but leave other types unchanged.

    Parameters
    ----------
    x : Any
        Value that may or may not be a string.
    **opts : Any
        Keyword arguments passed to text_normalize if x is a string.

    Returns
    -------
    Any
        Normalized string if x was a string, otherwise x unchanged.

    Examples
    --------
    >>> maybe_text_normalize("  HELLO  ", lowercase=True, strip=True)
    'hello'

    >>> maybe_text_normalize(42, lowercase=True)
    42
    """
    if isinstance(x, str):
        return text_normalize(x, **opts)
    return x


def maybe_text_normalize_with_options(
    x: Any,
    options: NormalizeOptions,
) -> Any:
    """Apply text normalization with structured options only if the value is a string.

    Parameters
    ----------
    x : Any
        Value that may or may not be a string.
    options : NormalizeOptions
        Normalization options to apply if x is a string.

    Returns
    -------
    Any
        Normalized string if x was a string, otherwise x unchanged.

    Examples
    --------
    >>> opts = NormalizeOptions()
    >>> maybe_text_normalize_with_options("  HELLO  ", opts)
    'hello'

    >>> maybe_text_normalize_with_options(42, opts)
    42
    """
    if isinstance(x, str):
        return text_normalize_with_options(x, options)
    return x


def fuzzy_match_score(
    s1: str,
    s2: str,
    algorithm: Literal[
        "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"
    ] = "token_set_ratio",
) -> float:
    """Calculate fuzzy similarity score between two strings.

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    algorithm : str, default "token_set_ratio"
        Fuzzy matching algorithm to use:
        - "ratio": Simple character-based similarity
        - "partial_ratio": Best matching substring
        - "token_sort_ratio": Token-based matching with sorting
        - "token_set_ratio": Token-based matching with set logic

    Returns
    -------
    float
        Similarity score between 0.0 and 1.0, where 1.0 is perfect match.

    Examples
    --------
    >>> fuzzy_match_score("hello world", "hello world")
    1.0

    >>> fuzzy_match_score("hello world", "hello wrld")  # Close match
    0.91...

    >>> fuzzy_match_score("apple", "orange")  # Poor match
    0.0
    """
    if algorithm == "ratio":
        score = fuzz.ratio(s1, s2)
    elif algorithm == "partial_ratio":
        score = fuzz.partial_ratio(s1, s2)
    elif algorithm == "token_sort_ratio":
        score = fuzz.token_sort_ratio(s1, s2)
    elif algorithm == "token_set_ratio":
        score = fuzz.token_set_ratio(s1, s2)
    else:
        raise ValueError(f"Unknown fuzzy algorithm: {algorithm}")

    # Convert from 0-100 to 0-1 scale
    return score / 100.0


def fuzzy_match_with_options(
    s1: str,
    s2: str,
    normalize_options: NormalizeOptions | None = None,
    fuzzy_options: FuzzyOptions | None = None,
) -> tuple[float, bool]:
    """Perform fuzzy matching with normalization and options.

    Normalization is always applied before fuzzy matching to improve
    match quality and consistency.

    Parameters
    ----------
    s1 : str
        First string to compare.
    s2 : str
        Second string to compare.
    normalize_options : NormalizeOptions | None, default None
        Normalization to apply before fuzzy matching. If None, defaults are used.
    fuzzy_options : FuzzyOptions | None, default None
        Fuzzy matching configuration. If None, defaults are used.

    Returns
    -------
    tuple[float, bool]
        Tuple of (similarity_score, is_match_above_threshold).
        similarity_score is 0.0-1.0, is_match_above_threshold is True if
        score >= threshold.

    Examples
    --------
    >>> # Default options: normalize + fuzzy matching
    >>> score, is_match = fuzzy_match_with_options("  Hello World  ", "hello world")
    >>> score  # Should be 1.0 due to normalization
    1.0
    >>> is_match  # Should be True
    True
    >>> # Custom threshold
    >>> opts = FuzzyOptions(threshold=0.9)
    >>> score, is_match = fuzzy_match_with_options("hello", "helo", fuzzy_options=opts)
    >>> score > 0.8 and score < 0.9  # Close but not exact
    True
    >>> is_match  # Should be False due to high threshold
    False
    """
    # Use defaults if options not provided
    if normalize_options is None:
        normalize_options = NormalizeOptions()
    if fuzzy_options is None:
        fuzzy_options = FuzzyOptions()

    # If fuzzy matching is disabled, do exact comparison after normalization
    if not fuzzy_options.enabled:
        norm_s1 = text_normalize_with_options(s1, normalize_options)
        norm_s2 = text_normalize_with_options(s2, normalize_options)
        score = 1.0 if norm_s1 == norm_s2 else 0.0
        is_match = score >= fuzzy_options.threshold
        return score, is_match

    # Apply normalization before fuzzy matching
    norm_s1 = text_normalize_with_options(s1, normalize_options)
    norm_s2 = text_normalize_with_options(s2, normalize_options)

    # Calculate fuzzy score
    score = fuzzy_match_score(norm_s1, norm_s2, fuzzy_options.algorithm)
    is_match = score >= fuzzy_options.threshold

    return score, is_match


def normalize_iter(
    it: Iterable[T],
    *,
    element_normalizer: Callable[[T], T] | None = None,
) -> list[T]:
    """Normalize elements in an iterable using a provided normalizer function.

    Parameters
    ----------
    it : Iterable[T]
        Iterable of values to normalize.
    element_normalizer : Callable[[T], T] | None, default None
        Function to apply to each element. If None, elements are returned as-is.

    Returns
    -------
    list[T]
        List of normalized elements.

    Examples
    --------
    >>> normalize_iter(["  APPLE  ", "BANANA!"],
    ...                element_normalizer=lambda x: x.strip().lower())
    ['apple', 'banana!']

    >>> normalize_iter([1, 2, 3])  # No normalizer
    [1, 2, 3]
    """
    if element_normalizer is None:
        return list(it)
    return [element_normalizer(v) for v in it]
