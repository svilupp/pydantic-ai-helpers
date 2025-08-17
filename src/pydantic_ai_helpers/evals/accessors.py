"""Safe dotted-path accessors for evaluation data.

This module provides utilities for safely accessing nested data structures
using dotted path notation, with robust error handling and informative
error messages.

Examples
--------
    >>> obj = {"user": {"name": "Alice", "scores": [10, 20, 30]}}
    >>> ok, value, reason = resolve_path(obj, "user.name")
    >>> print(ok, value)  # True, "Alice"

    >>> ok, value, reason = resolve_path(obj, "user.scores.1")
    >>> print(ok, value)  # True, 20

    >>> ok, value, reason = resolve_path(obj, "user.missing")
    >>> # False, "Path 'user.missing': missing key/attr 'missing' on ..."
    >>> print(ok, reason)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_SENTINEL = object()


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


def _segment_iter(path: str) -> Iterable[str]:
    """Split a dotted path into segments.

    Parameters
    ----------
    path : str
        Dotted path string (e.g., "user.scores.0").

    Yields
    ------
    str
        Individual path segments.
    """
    if not path:
        return
    for seg in path.split("."):
        if seg.strip():
            yield seg.strip()


def resolve_path(root: Any, path: str | None) -> tuple[bool, Any, str | None]:
    """Safely resolve a dotted path into the root object.

    Supports dict keys, object attributes, and sequence indices.
    Never raises exceptions - returns error information instead.

    Parameters
    ----------
    root : Any
        Root object to traverse.
    path : str | None
        Dotted path to resolve (e.g., "user.scores.0").
        None or empty string returns the root object.

    Returns
    -------
    tuple[bool, Any, str | None]
        Tuple of (success, value, error_reason).
        If success is True, value contains the resolved object and error_reason is None.
        If success is False, value is None and error_reason contains an error message.

    Examples
    --------
    >>> obj = {"user": {"name": "Alice", "scores": [10, 20]}}
    >>> ok, value, reason = resolve_path(obj, "user.name")
    >>> print(ok, value)  # True, "Alice"

    >>> ok, value, reason = resolve_path(obj, "user.scores.1")
    >>> print(ok, value)  # True, 20

    >>> ok, value, reason = resolve_path(obj, "missing")
    >>> print(ok, reason)  # False, "Path 'missing': missing key/attr 'missing' on ..."
    """
    if path in (None, "", "."):
        return True, root, None

    cur = root
    for seg in _segment_iter(path):
        if cur is None:
            return False, None, f"Path '{path}': segment '{seg}' on None"

        # Check if segment is a numeric index
        idx = None
        if seg.isdigit() or (seg.startswith("-") and seg[1:].isdigit()):
            idx = int(seg)

        try:
            if idx is not None:  # sequence access
                if not hasattr(cur, "__getitem__"):
                    return (
                        False,
                        None,
                        f"Path '{path}': segment '{seg}' requires indexable type, "
                        f"got {type(cur).__name__}",
                    )
                cur = cur[idx]
                continue

            # dict key access
            if isinstance(cur, dict) and seg in cur:
                cur = cur[seg]
                continue

            # attribute access
            if hasattr(cur, seg):
                cur = getattr(cur, seg)
                continue

            # No valid access method found
            return (
                False,
                None,
                f"Path '{path}': missing key/attr '{seg}' on {_repr_trunc(cur)}",
            )

        except Exception as e:  # pragma: no cover
            return False, None, f"Path '{path}': error at '{seg}': {e!r}"

    return True, cur, None


@dataclass(frozen=True)
class Accessor:
    """Reusable accessor wrapper for safe path resolution.

    This provides a convenient way to package path resolution logic
    for reuse across multiple evaluations.

    Parameters
    ----------
    path : str | None, default None
        Dotted path to resolve. None means "use value as-is".

    Examples
    --------
    >>> accessor = Accessor("user.name")
    >>> obj = {"user": {"name": "Alice"}}
    >>> ok, value, reason = accessor.get(obj)
    >>> print(ok, value)  # True, "Alice"
    """

    path: str | None = None

    def get(self, obj: Any) -> tuple[bool, Any, str | None]:
        """Resolve the configured path on the given object.

        Parameters
        ----------
        obj : Any
            Object to resolve the path on.

        Returns
        -------
        tuple[bool, Any, str | None]
            Result of path resolution. See resolve_path for details.
        """
        return resolve_path(obj, self.path)
