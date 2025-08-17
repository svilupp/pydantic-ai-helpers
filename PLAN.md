Below is a complete, implementation‑ready proposal to add **Pydantic Evals** utilities to `pydantic-ai-helpers`. It’s organized so you can copy files in, run tests, and immediately start writing compact, readable evals.

I’ve first summarized how `pydantic-evals` wants evaluators and results to look (so we match the API precisely), then I present the package design, detailed code (with docstrings), examples, and a test plan. I also include migration/packaging notes and documentation copy.

---

## 0) What Pydantic Evals expects (so we fit perfectly)

* **Evaluator/EvaluatorContext**: An evaluator is a small dataclass with an `evaluate(ctx)` method. The `ctx` includes `inputs`, `output`, `expected_output`, duration, attributes, and more. We’ll use these fields extensively to extract, compare, and grade. ([Pydantic AI][1])
* **Return type**: `evaluate()` may return a scalar (bool/int/float/str), or `EvaluationReason` (with `value` and `reason`), or a mapping of names to either of those. We’ll standardize on returning `EvaluationReason` for rich traceability. ([Pydantic AI][1])
* **EvaluationReason** is the “specialized output grade” you described: it carries a scalar `value` plus an optional human‑readable `reason`. We’ll always populate `reason`. ([Pydantic AI][2])

Example built‑ins such as `IsInstance`, `Equals`, `Contains`, `MaxDuration` etc. are implemented as simple dataclasses, which we’ll emulate. ([Pydantic AI][1])

---

## 1) What we’ll add to **pydantic-ai-helpers**

We’ll add a new subpackage focused on **field extraction and robust comparisons** that compile down to Pydantic Evals evaluators. The philosophy: tiny, composable building blocks + ready‑made evaluators for the most common patterns you described.

**Package layout (new):**

```
src/pydantic_ai_helpers/evals/
    __init__.py
    accessors.py         # Safe dotted-path getters and helpers
    normalize.py         # Shared text/sequence normalization
    compare.py           # Scalar, list, inclusion comparators (pure Python)
    evaluators.py        # Reusable Evaluator implementations built on compare.py
    registry.py          # Tiny helpers to register evaluators on a Dataset
```

> This aligns with how `pydantic-evals` models evaluators (simple dataclasses), where we’ll primarily return `EvaluationReason` to capture “value + reason”. ([Pydantic AI][2])

---

## 2) Feature design (maps 1‑to‑1 to your needs)

### 2.1 Safe, dotted‑path accessors

* **Goal**: `"a.b.c"` means “safely get `.a` then `.b` then `.c`” from a Python object that might be a dict, a dataclass/Pydantic model, or an object with attributes. Numeric path segments address sequences, e.g. `"items.0.name"`.
* **Failure**: Missing field ⇒ automatic failure in the evaluator with an informative reason.
* **API**:

  * `resolve_path(obj, path: str) -> tuple[bool, Any, str | None]`
  * Supports: dict key, attribute, sequence index, nested arbitrarily.
  * Never raises on missing; returns `(False, None, reason)`.

### 2.2 Normalization options (reusable across comparators)

Per‑value transform options:

* `lowercase: bool`
* `strip: bool`
* `alphanum: bool` (remove all non‑\[A-Za-z0-9])
* `collapse_spaces: bool` (optional but often handy)

Applied to elements of lists or to scalars. This gives the consistent “lowercase/strip/alphanum” behavior you asked for.

### 2.3 Comparators

All comparators return a `(value, reason)` pair for easy `EvaluationReason` construction.

* **ScalarCompare**

  * **Type coercion**: attempt to coerce both sides to a target type: `"str" | "int" | "float" | "bool" | Enum`.
  * **Enums**: compare by name (string) after normalization; you can pass `enum_values` as names or actual Enum class.
  * **Numbers**: support `abs_tol` and `rel_tol` — either or both.
  * **Strings**: optional case/strip/alphanum (via `normalize`).
  * Failure modes produce precise reasons (e.g., “failed to coerce ‘abc’ to float” or “mismatch by 0.13 (abs\_tol=0.1)”).

* **ListCompare**

  * **Modes**:

    * `equality` (order‑sensitive or not),
    * `recall` = |intersection| / |expected|,
    * `precision` = |intersection| / |output|.
  * **Set vs multiset**: default set semantics (unique elements), optional `multiset=True`.
  * **Element equivalence**: apply normalization and (optionally) scalar coercion to each element.
  * **Edge cases** documented:

    * `expected == []`: recall = 1.0 by convention if output is also empty, otherwise 1.0 (no requirements) — we’ll make this explicit in reasons so teams can change it later.
    * `output == []`: precision = 1.0 if expected empty else 0.0.

* **InclusionCompare**

  * Single value vs list: is `output_value` in `expected_list` (after normalization/coercion)?
  * Also covers string vs list of synonyms like “cola” ∈ \[“coke”, “cola”, “coca-cola”].

### 2.4 Reusable Evaluators

We ship a small family:

1. **`CompareFields`** — the core, generic evaluator.

   * Parameters:

     * `output_path: str | None` (defaults to root)
     * `expected_path: str | None` (defaults to root)
     * `comparator: Comparator` (one of the comparators above)
     * `evaluation_name: str | None` (useful in reports)
   * Behavior: extract `left = get(ctx.output, output_path)`, `right = get(ctx.expected_output, expected_path)`, compare with comparator and return `EvaluationReason`.

2. **`ScalarEquals`** — sugar over `CompareFields` with `ScalarCompare(coerce_to=..., abs_tol=..., rel_tol=...)`.

3. **`ListEquality` / `ListRecall` / `ListPrecision`** — sugar over `CompareFields` with `ListCompare(mode=...)`.

4. **`ValueInExpectedList`** — sugar over `CompareFields` with `InclusionCompare`.

5. **`MultiCompare`** — aggregate multiple field comparisons with optional weights to compute an overall score and reason roll‑up (we’ll return per‑field reasons in a JSON-ish reason text plus the weighted mean score).

> All of these are plain `@dataclass` subclasses of `Evaluator`, returning `EvaluationReason`. This mirrors the builtin examples and API docs. ([Pydantic AI][1])

### 2.5 Convenience registration helpers

* `register(dataset, *evaluators)` to append ours to a dataset.
* `from_specs(dataset, list_of_specs)` to build and register `CompareFields` et al. from compact dicts (nice for YAML/JSON eval suites).

> Datasets accept evaluators and expose results compatible with `EvaluationResult`. We’ll make evaluator `evaluation_name` match how `EvaluationResult.name` is displayed. ([PyPI][3])

---

## 3) Implementation (drop-in code)

> **Note**: these snippets are designed to live under `src/pydantic_ai_helpers/evals/`. They avoid any heavy dependencies and follow the `pydantic-evals` dataclass evaluator style. Where helpful, I truncate value reprs to keep “reasons” readable.

### 3.1 `accessors.py`

```python
# src/pydantic_ai_helpers/evals/accessors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

_SENTINEL = object()

def _repr_trunc(value: Any, max_len: int = 120) -> str:
    r = repr(value)
    return r if len(r) <= max_len else r[: max_len - 1] + "…"

def _segment_iter(path: str) -> Iterable[str]:
    # simple dotted path; supports numeric segments for sequence indices
    if not path:
        return ()
    return (seg.strip() for seg in path.split(".") if seg.strip())

def resolve_path(root: Any, path: str | None):
    """
    Safely resolve a dotted path into `root`.
    Returns: (ok: bool, value: Any, reason: str | None)
    """
    if path in (None, "", "."):
        return True, root, None

    cur = root
    for seg in _segment_iter(path):
        if cur is None:
            return False, None, f"Path '{path}': segment '{seg}' on None"
        # sequence index?
        idx = None
        if seg.isdigit() or (seg.startswith("-") and seg[1:].isdigit()):
            idx = int(seg)

        try:
            if idx is not None:  # sequence
                if not hasattr(cur, "__getitem__"):
                    return False, None, f"Path '{path}': segment '{seg}' requires indexable type, got {type(cur).__name__}"
                cur = cur[idx]
                continue

            # dict key?
            if isinstance(cur, dict) and seg in cur:
                cur = cur[seg]
                continue

            # attribute?
            if hasattr(cur, seg):
                cur = getattr(cur, seg)
                continue

            # pydantic v2 models expose attributes; above should handle it
            # fallback?
            return False, None, f"Path '{path}': missing key/attr '{seg}' on {_repr_trunc(cur)}"
        except Exception as e:  # pragma: no cover (defensive)
            return False, None, f"Path '{path}': error at '{seg}': {e!r}"

    return True, cur, None

@dataclass(frozen=True)
class Accessor:
    """Reusable accessor wrapper for output/expected.
    `path=None` means 'use value as-is'.
    """
    path: str | None = None

    def get(self, obj: Any):
        return resolve_path(obj, self.path)
```

### 3.2 `normalize.py`

```python
# src/pydantic_ai_helpers/evals/normalize.py
from __future__ import annotations

import re
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")

_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")

def text_normalize(
    s: str,
    *,
    lowercase: bool = False,
    strip: bool = False,
    alphanum: bool = False,
    collapse_spaces: bool = False,
) -> str:
    if strip:
        s = s.strip()
    if lowercase:
        s = s.lower()
    if alphanum:
        s = _ALNUM_RE.sub("", s)
    if collapse_spaces:
        s = " ".join(s.split())
    return s

def maybe_text_normalize(
    x: Any,
    **opts: Any,
) -> Any:
    if isinstance(x, str):
        return text_normalize(x, **opts)
    return x

def normalize_iter(
    it: Iterable[T],
    *,
    element_normalizer: Callable[[T], T] | None = None,
) -> list[T]:
    if element_normalizer is None:
        return list(it)
    return [element_normalizer(v) for v in it]
```

### 3.3 `compare.py`

```python
# src/pydantic_ai_helpers/evals/compare.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Type, cast
from math import isfinite, isclose

from .normalize import maybe_text_normalize, normalize_iter

def _repr_trunc(value: Any, max_len: int = 120) -> str:
    r = repr(value)
    return r if len(r) <= max_len else r[: max_len - 1] + "…"

# ---------- type coercion ----------

class CoercionError(ValueError): ...

def _coerce_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
    raise CoercionError(f"cannot coerce {x!r} to bool")

def _coerce(x: Any, to: str | Type[object], *, enum_values: Sequence[str] | None = None) -> Any:
    if isinstance(to, str):
        kind = to
    elif isinstance(to, type):
        # allow Enum class passed in; compare by name
        if hasattr(to, "__members__"):  # Enum
            enum_values = list(getattr(to, "__members__").keys())
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
        # compare by string identity; case-insensitive
        s_norm = s.strip().lower()
        for name in enum_values:
            if s_norm == name.lower():
                return name  # normalized enum member name
        raise CoercionError(f"{x!r} not in enum {sorted(enum_values)!r}")
    raise CoercionError(f"unsupported coercion target {to!r}")

# ---------- comparators ----------

@dataclass(frozen=True)
class ScalarCompare:
    """Compare two scalar values.

    Args:
        coerce_to: "str"|"int"|"float"|"bool"|"enum"|type (Enum class)
        abs_tol: numeric absolute tolerance (for float/int)
        rel_tol: numeric relative tolerance (for float comparisons)
        normalize_opts: kwargs for string normalization (lowercase/strip/alphanum/collapse_spaces)
        enum_values: names to accept if coerce_to="enum" (or pass an Enum class as coerce_to)
    """
    coerce_to: str | type | None = None
    abs_tol: float | None = None
    rel_tol: float | None = None
    normalize_opts: Mapping[str, Any] | None = None
    enum_values: Sequence[str] | None = None

    def __call__(self, left: Any, right: Any) -> tuple[Any, str]:
        # Normalize text before coercion
        l = maybe_text_normalize(left, **(self.normalize_opts or {}))
        r = maybe_text_normalize(right, **(self.normalize_opts or {}))

        # Attempt coercion
        if self.coerce_to is not None:
            try:
                l = _coerce(l, self.coerce_to, enum_values=self.enum_values)
            except Exception as e:
                return 0.0, f"left coercion failed: {e}"
            try:
                r = _coerce(r, self.coerce_to, enum_values=self.enum_values)
            except Exception as e:
                return 0.0, f"right coercion failed: {e}"

        # Numbers with tolerance
        if isinstance(l, (int, float)) and isinstance(r, (int, float)):
            if not (isfinite(float(l)) and isfinite(float(r))):
                return 0.0, f"non-finite number(s): left={l!r}, right={r!r}"
            abs_tol = self.abs_tol if self.abs_tol is not None else 0.0
            rel_tol = self.rel_tol if self.rel_tol is not None else 0.0
            ok = isclose(float(l), float(r), rel_tol=rel_tol, abs_tol=abs_tol)
            value = 1.0 if ok else 0.0
            reason = "numbers match" if ok else f"numbers differ: {l!r} vs {r!r} (abs_tol={abs_tol}, rel_tol={rel_tol})"
            return value, reason

        # Everything else: exact equality after normalization/coercion
        ok = (l == r)
        return (1.0 if ok else 0.0, "values equal" if ok else f"values differ: { _repr_trunc(l) } vs { _repr_trunc(r) }")

@dataclass(frozen=True)
class ListCompare:
    """Compare two sequences.

    Args:
        mode: "equality" | "recall" | "precision"
        order_sensitive: for equality mode only
        multiset: if True, count duplicates in precision/recall/equality; default is set semantics
        normalize_opts: normalize elements (strings only) before comparing
        element_coerce_to: optional scalar coercion for elements
    """
    mode: str = "equality"
    order_sensitive: bool = False
    multiset: bool = False
    normalize_opts: Mapping[str, Any] | None = None
    element_coerce_to: str | type | None = None

    def _prep(self, xs: Iterable[Any]) -> list[Any]:
        xs = [maybe_text_normalize(x, **(self.normalize_opts or {})) for x in xs]
        if self.element_coerce_to is not None:
            out = []
            for x in xs:
                try:
                    out.append(_coerce(x, self.element_coerce_to))
                except Exception:
                    # Keep uncoercible as-is to make mismatch visible in reason
                    out.append(x)
            return out
        return xs

    def _to_bag(self, xs: Iterable[Any]) -> Mapping[Any, int]:
        from collections import Counter
        return Counter(xs)

    def __call__(self, left: Any, right: Any) -> tuple[Any, str]:
        if not isinstance(left, Iterable) or isinstance(left, (str, bytes)):
            return 0.0, f"left is not a sequence: {type(left).__name__}"
        if not isinstance(right, Iterable) or isinstance(right, (str, bytes)):
            return 0.0, f"right is not a sequence: {type(right).__name__}"

        L = self._prep(list(left))
        R = self._prep(list(right))

        if self.mode == "equality":
            if self.multiset:
                ok = self._to_bag(L) == self._to_bag(R)
            else:
                ok = (L == R) if self.order_sensitive else (set(L) == set(R))
            return (1.0 if ok else 0.0, "lists equal" if ok else f"lists differ: left={_repr_trunc(L)}, right={_repr_trunc(R)}")

        # set/multiset math for precision/recall
        if self.multiset:
            # multiset intersection size
            from collections import Counter
            cL, cR = Counter(L), Counter(R)
            inter = sum((cL & cR).values())
            if self.mode == "recall":
                denom = sum(cR.values())
            else:  # precision
                denom = sum(cL.values())
        else:
            sL, sR = set(L), set(R)
            inter = len(sL & sR)
            denom = len(sR) if self.mode == "recall" else len(sL)

        if denom == 0:
            # convention: no requirements -> perfect score
            return 1.0, f"{self.mode}: denominator=0 (no requirements)"

        score = inter / denom
        return score, f"{self.mode}: hits={inter}, denom={denom}, score={score:.4f}"

@dataclass(frozen=True)
class InclusionCompare:
    """Single value 'left' is included in sequence 'right' (exact match after normalization/coercion)."""
    normalize_opts: Mapping[str, Any] | None = None
    element_coerce_to: str | type | None = None

    def __call__(self, left: Any, right: Any) -> tuple[Any, str]:
        l = maybe_text_normalize(left, **(self.normalize_opts or {}))
        try:
            if self.element_coerce_to is not None:
                l = _coerce(l, self.element_coerce_to)
        except Exception as e:
            return 0.0, f"left coercion failed: {e}"

        if not isinstance(right, Iterable) or isinstance(right, (str, bytes)):
            return 0.0, f"right is not a sequence: {type(right).__name__}"

        elems = []
        for x in right:
            x_norm = maybe_text_normalize(x, **(self.normalize_opts or {}))
            try:
                if self.element_coerce_to is not None:
                    x_norm = _coerce(x_norm, self.element_coerce_to)
            except Exception:
                pass
            elems.append(x_norm)

        ok = any(e == l for e in elems)
        return (1.0 if ok else 0.0, f"{_repr_trunc(l)} {'in' if ok else 'not in'} {_repr_trunc(elems)}")
```

### 3.4 `evaluators.py`

```python
# src/pydantic_ai_helpers/evals/evaluators.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason  # type: ignore
# API docs confirm Evaluator/EvaluatorContext/EvaluationReason names and semantics. :contentReference[oaicite:7]{index=7}

from .accessors import Accessor
from .compare import ScalarCompare, ListCompare, InclusionCompare, _repr_trunc

def _reason_prefix(name: str | None) -> str:
    return f"[{name}] " if name else ""

@dataclass(repr=False)
class CompareFields(Evaluator[object, object, object]):
    """Generic field-to-field comparison.

    Extracts values using dotted paths from ctx.output and ctx.expected_output, then
    compares them with the provided comparator.

    Returns an EvaluationReason with score/boolean in `value` and a human-friendly `reason`.
    """
    output_path: str | None = None
    expected_path: str | None = None
    comparator: Any = None  # callable (left, right) -> (value, reason)
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        # resolve output
        ok_l, left, r_l = Accessor(self.output_path).get(ctx.output)
        if not ok_l:
            return EvaluationReason(value=0.0, reason=_reason_prefix(self.evaluation_name) + f"output path error: {r_l}")

        # resolve expected
        if ctx.expected_output is None:
            return EvaluationReason(value=0.0, reason=_reason_prefix(self.evaluation_name) + "expected_output missing")

        ok_r, right, r_r = Accessor(self.expected_path).get(ctx.expected_output)
        if not ok_r:
            return EvaluationReason(value=0.0, reason=_reason_prefix(self.evaluation_name) + f"expected path error: {r_r}")

        # compare
        if self.comparator is None:
            return EvaluationReason(value=0.0, reason=_reason_prefix(self.evaluation_name) + "no comparator provided")

        value, why = self.comparator(left, right)
        return EvaluationReason(value=value, reason=_reason_prefix(self.evaluation_name) + why)

# ---- Specializations ----

@dataclass(repr=False)
class ScalarEquals(CompareFields):
    coerce_to: str | type | None = None
    abs_tol: float | None = None
    rel_tol: float | None = None
    normalize_opts: Mapping[str, Any] | None = None
    enum_values: Iterable[str] | None = None

    def __post_init__(self):
        self.comparator = ScalarCompare(
            coerce_to=self.coerce_to,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
            normalize_opts=self.normalize_opts,
            enum_values=list(self.enum_values) if self.enum_values is not None else None,
        )

@dataclass(repr=False)
class ListEquality(CompareFields):
    order_sensitive: bool = False
    multiset: bool = False
    normalize_opts: Mapping[str, Any] | None = None
    element_coerce_to: str | type | None = None

    def __post_init__(self):
        self.comparator = ListCompare(
            mode="equality",
            order_sensitive=self.order_sensitive,
            multiset=self.multiset,
            normalize_opts=self.normalize_opts,
            element_coerce_to=self.element_coerce_to,
        )

@dataclass(repr=False)
class ListRecall(CompareFields):
    multiset: bool = False
    normalize_opts: Mapping[str, Any] | None = None
    element_coerce_to: str | type | None = None

    def __post_init__(self):
        self.comparator = ListCompare(
            mode="recall",
            multiset=self.multiset,
            normalize_opts=self.normalize_opts,
            element_coerce_to=self.element_coerce_to,
        )

@dataclass(repr=False)
class ListPrecision(CompareFields):
    multiset: bool = False
    normalize_opts: Mapping[str, Any] | None = None
    element_coerce_to: str | type | None = None

    def __post_init__(self):
        self.comparator = ListCompare(
            mode="precision",
            multiset=self.multiset,
            normalize_opts=self.normalize_opts,
            element_coerce_to=self.element_coerce_to,
        )

@dataclass(repr=False)
class ValueInExpectedList(CompareFields):
    normalize_opts: Mapping[str, Any] | None = None
    element_coerce_to: str | type | None = None

    def __post_init__(self):
        self.comparator = InclusionCompare(
            normalize_opts=self.normalize_opts,
            element_coerce_to=self.element_coerce_to,
        )

# ---- Aggregating multiple comparisons ----

@dataclass(repr=False)
class MultiCompare(Evaluator[object, object, object]):
    """Run multiple CompareFields (or compatible) and aggregate.

    `specs` is a list of CompareFields instances (or subclasses).
    `weights` optional, same length as specs; default = 1.0 each.

    Returns EvaluationReason with weighted mean score in value and a compact JSON reason.
    """
    specs: list[CompareFields]
    weights: list[float] | None = None
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        if not self.specs:
            return EvaluationReason(value=0.0, reason=_reason_prefix(self.evaluation_name) + "no specs provided")

        w = self.weights or [1.0] * len(self.specs)
        if len(w) != len(self.specs):
            return EvaluationReason(value=0.0, reason=_reason_prefix(self.evaluation_name) + "weights length mismatch")

        parts: list[tuple[str, float, str | None, str]] = []
        total, wsum = 0.0, 0.0
        for spec, wi in zip(self.specs, w, strict=False):
            out = spec.evaluate(ctx)
            name = spec.evaluation_name or type(spec).__name__
            value = float(out.value) if isinstance(out.value, (int, float)) else (1.0 if out.value else 0.0)
            total += wi * value
            wsum += wi
            parts.append((name, value, out.reason, spec.output_path or "."))

        score = total / (wsum or 1.0)
        # synthesize a readable reason payload
        detail = {
            "overall": round(score, 6),
            "components": [
                {"name": n, "score": round(v, 6), "path": p, "reason": r}
                for (n, v, r, p) in parts
            ],
        }
        return EvaluationReason(
            value=score,
            reason=_reason_prefix(self.evaluation_name) + _repr_trunc(detail, max_len=500),
        )
```

### 3.5 `registry.py`

```python
# src/pydantic_ai_helpers/evals/registry.py
from __future__ import annotations

from typing import Any, Iterable

def register(dataset: Any, *evaluators: Any) -> None:
    """Append evaluators to a `pydantic_evals.Dataset`-like object."""
    for ev in evaluators:
        dataset.add_evaluator(ev)  # API shown in Pydantic Evals docs and PyPI page. :contentReference[oaicite:8]{index=8}

def from_specs(dataset: Any, specs: Iterable[dict[str, Any]]) -> None:
    """Build CompareFields etc. from compact dict specs and register them.

    Spec example:
      {
        "kind": "ListRecall",
        "name": "recall:colors",
        "output_path": "predicted.colors",
        "expected_path": "gold.colors",
        "normalize": {"lowercase": True, "strip": True}
      }
    """
    from .evaluators import (
        CompareFields, ListRecall, ListPrecision, ListEquality,
        ScalarEquals, ValueInExpectedList
    )

    kind_map = {
        "CompareFields": CompareFields,
        "ListRecall": ListRecall,
        "ListPrecision": ListPrecision,
        "ListEquality": ListEquality,
        "ScalarEquals": ScalarEquals,
        "ValueInExpectedList": ValueInExpectedList,
    }

    for s in specs:
        k = s.pop("kind")
        cls = kind_map[k]
        name = s.pop("name", None)
        normalize = s.pop("normalize", None)
        if normalize is not None:
            s["normalize_opts"] = normalize
        ev = cls(**s)
        if name is not None:
            setattr(ev, "evaluation_name", name)
        dataset.add_evaluator(ev)
```

### 3.6 `__init__.py`

```python
# src/pydantic_ai_helpers/evals/__init__.py
from .accessors import Accessor, resolve_path
from .normalize import text_normalize
from .compare import ScalarCompare, ListCompare, InclusionCompare
from .evaluators import (
    CompareFields,
    ScalarEquals,
    ListEquality,
    ListRecall,
    ListPrecision,
    ValueInExpectedList,
    MultiCompare,
)
from .registry import register, from_specs

__all__ = [
    "Accessor", "resolve_path", "text_normalize",
    "ScalarCompare", "ListCompare", "InclusionCompare",
    "CompareFields", "ScalarEquals", "ListEquality", "ListRecall", "ListPrecision",
    "ValueInExpectedList", "MultiCompare", "register", "from_specs",
]
```

---

## 4) Usage examples

### 4.1 Small, direct evaluator

```python
from pydantic_evals import Dataset, Case
from pydantic_ai_helpers.evals import ScalarEquals, ListRecall, ValueInExpectedList

cases = [
    Case(
        name="bev",
        inputs={"q": "Pick a cola"},
        expected_output={"canonical": "cola", "allowed": ["coke", "cola", "coca-cola"]},
    )
]

dataset = Dataset(cases=cases)

# output the model returns (for example):
model_output = {"canonical": "Cola", "predicted": ["Cola", "Pepsi"]}

# Compare canonical (scalar, string-insensitive)
dataset.add_evaluator(
    ScalarEquals(
        evaluation_name="canonical-eq",
        output_path="canonical",
        expected_path="canonical",
        coerce_to="str",
        normalize_opts={"lowercase": True, "strip": True},
    )
)

# Inclusion: canonical appears in expected allowed list
dataset.add_evaluator(
    ValueInExpectedList(
        evaluation_name="canonical-in-allowed",
        output_path="canonical",
        expected_path="allowed",
        normalize_opts={"lowercase": True, "strip": True},
    )
)

# Recall on predicted vs expected allowed
dataset.add_evaluator(
    ListRecall(
        evaluation_name="recall:allowed",
        output_path="predicted",
        expected_path="allowed",
        normalize_opts={"lowercase": True, "strip": True},
    )
)

# Evaluate a function that returns model_output:
def task(_inputs: dict) -> dict:
    return model_output

report = dataset.evaluate_sync(task)
report.print(include_input=True, include_output=True)
```

> These evaluators return `EvaluationReason` with a numeric value and an explanation reason, matching the `pydantic-evals` interface. ([Pydantic AI][2])

### 4.2 Multi‑field spec with aggregation

```python
from pydantic_evals import Dataset, Case
from pydantic_ai_helpers.evals import (
    ListRecall, ScalarEquals, MultiCompare, register
)

cases = [
    Case(
        name="colors",
        inputs=None,
        expected_output={"names": ["blue", "green"], "count": 2},
    )
]
dataset = Dataset(cases=cases)

cmp1 = ListRecall(
    evaluation_name="names-recall",
    output_path="names", expected_path="names",
    normalize_opts={"lowercase": True}
)
cmp2 = ScalarEquals(
    evaluation_name="count-exact",
    output_path="count", expected_path="count",
    coerce_to="int"
)

register(dataset, MultiCompare(specs=[cmp1, cmp2], weights=[0.7, 0.3], evaluation_name="overall"))

def task(_):
    return {"names": ["Green", "Blue"], "count": 2}

dataset.evaluate_sync(task).print()
```

---

## 5) Why this shape? (decisions & reasoning)

* **Layered**: We provide small, reusable comparators plus immediate, ergonomic evaluators (`ScalarEquals`, `ListRecall`, …). You can either (a) compose your own evaluator quickly or (b) use the ready‑made ones. This mirrors how the core library offers general `Evaluator` mechanics and a few convenience evaluators. ([Pydantic AI][1])
* **Always `EvaluationReason`**: you asked for “value + reason” everywhere to explain failures and guide debugging—exactly the point of `EvaluationReason`. ([Pydantic AI][2])
* **Strict field access**: missing fields cause immediate failure with precise reasons (“output path error”, “expected path error”). This makes silent data issues obvious.
* **Normalization pipeline**: the `lowercase`/`strip`/`alphanum` knobs apply uniformly to scalars and list elements; one shared implementation reduces bugs.
* **Type coercion**: comparators attempt conversions first (as you requested) and fail cleanly if impossible. Numbers support `abs_tol` and `rel_tol`. Enums compare by member names (strings), case‑insensitive for robustness.
* **List math**: recall/precision are explained with “hits/denom/score” so you can see what happened.
* **Serialization friendliness**: evaluators are dataclasses; `pydantic-evals` will serialize them into `EvaluatorSpec` cleanly for reports. (It gets the name from the class or the `evaluation_name` attribute.) ([Pydantic AI][1])

---

## 6) Test plan (pytest)

> The repo aims for high coverage; the tests below cover core branches. Add them under `tests/evals/`.

**`test_accessors.py`**

* dict, attr, pydantic model, dataclass, sequence index
* missing keys/attrs and None behavior
* negative index support

**`test_normalize.py`**

* lowercase/strip/alphanum/collapse\_spaces combos

**`test_scalar_compare.py`**

* str==str with normalization
* int/float coercion, abs/rel tolerance edges
* bool coercion variants (“Y/Yes/True/1”, “No/0”, etc.)
* enum by names, bad enum value
* failed coercion yields 0 with reason

**`test_list_compare.py`**

* equality (order sensitive/insensitive, set vs multiset)
* recall and precision with varied overlaps
* normalize on elements
* empty denom conventions

**`test_inclusion_compare.py`**

* value in list (with normalization/coercion)
* not in list, detailed reason

**`test_evaluators.py`**

* CompareFields with both paths present
* missing output path / missing expected\_output / expected path error
* specializations: ScalarEquals/ListRecall/ListPrecision/ListEquality/ValueInExpectedList
* MultiCompare weights, reason JSON integrity

**`test_integration_dataset.py`**

* Construct a tiny `Dataset` with a `Case`, add our evaluators via `registry.register`, run `evaluate_sync`, assert values & reasons shape (don’t assert exact text, assert substrings).

---

## 7) Developer docs (to add to your site/README)

### New section: *Evals helpers*

> **Install**
>
> ```bash
> pip install pydantic-ai-helpers pydantic-evals
> ```
>
> **Why use these?** You often want to compare *fields* of your output to *fields* of your expected output, not just entire objects. Our helpers give you:
>
> * Safe dotted‑path lookup (e.g., `"predictions.0.label"`).
> * Shared normalization knobs (`lowercase`, `strip`, `alphanum`) for scalars and lists.
> * Ready‑made list metrics (`recall`, `precision`, and strict equality).
> * Numerical comparisons with tolerances.
> * Always returns `EvaluationReason(value, reason)` for transparent debugging. ([Pydantic AI][2])

**Quick example**

```python
from pydantic_evals import Case, Dataset
import pydantic_ai_helpers.evals as phe

cases = [Case(inputs=None, expected_output={"ids": ["a", "b", "c"]})]
dataset = Dataset(cases=cases)

dataset.add_evaluator(
    phe.ListRecall(
        evaluation_name="ids-recall",
        output_path="ids",
        expected_path="ids",
        normalize_opts={"lowercase": True}
    )
)
```

**API cheatsheet**

* **Accessors**:

  * `resolve_path(obj, "a.b.0.c") -> (ok, value, reason)`
* **Comparators**:

  * `ScalarCompare(coerce_to="float", abs_tol=0.01)`
  * `ListCompare(mode="recall", multiset=False, normalize_opts={...})`
  * `InclusionCompare(normalize_opts={...})`
* **Evaluators**:

  * `CompareFields(output_path="...", expected_path="...", comparator=...)`
  * `ScalarEquals(...)`, `ListEquality(...)`, `ListRecall(...)`, `ListPrecision(...)`, `ValueInExpectedList(...)`
  * `MultiCompare(specs=[...], weights=[...])`

---

## 8) Packaging & compatibility notes

* Add dependency in your `pyproject.toml`:

  ```toml
  [project.optional-dependencies]
  evals = ["pydantic-evals>=0.7"]  # latest public releases as of Aug 2025
  ```

  (Pin broadly compatible with current docs and release cadence.) ([PyPI][3])
* Keep the new helpers **optional** so users who don’t run evals don’t pull extra deps.
* Our evaluators are pure `@dataclass` classes that comply with the `Evaluator` API from `pydantic-evals`, including `evaluation_name` for nice report labels. ([Pydantic AI][1])

---

## 9) Ready‑to‑use “MyEvaluator” example (from your snippet)

You can still write custom evaluators trivially; they coexist with these utilities:

```python
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason

@dataclass
class MyEvaluator(Evaluator[str, str, object]):
    def evaluate(self, ctx: EvaluatorContext[str, str]) -> EvaluationReason:
        if ctx.output == ctx.expected_output:
            return EvaluationReason(1.0, "exact match")
        if isinstance(ctx.output, str) and ctx.expected_output and ctx.expected_output.lower() in ctx.output.lower():
            return EvaluationReason(0.8, "substring match")
        return EvaluationReason(0.0, "no match")
```

This uses `EvaluationReason(value, reason)` as recommended by the API. ([Pydantic AI][2])

---

## 10) Roadmap / future extensions (not required now)

* **JSONPath/JMESPath adapters**: behind a small optional extra (e.g., `evals-jsonpath`) for more expressive selectors, keeping the default dotted‑path behavior fast and zero‑extra‑deps.
* **Fuzzy string comparators**: e.g., token Jaccard, Levenshtein thresholds — pair neatly with our normalization knobs.
* **Field discovery utilities**: print which dotted paths exist inside `ctx.output` to aid authoring new evals.
* **Span‑aware timing/telemetry checks**: additional evaluators that inspect `ctx.span_tree` when Logfire is enabled (the context exposes a span tree property). ([Pydantic AI][1])

---

## 11) How this integrates with the existing library

Your helpers currently focus on **history navigation** and conversation utilities (e.g., `History`, `ToolsView`, `MediaView`), which complements these eval helpers nicely: you can extract what you need with `History`, then write evaluators that check structured outputs. I kept the style “boring, explicit, type‑safe” to match the project’s philosophy on your docs site. ([siml.earth][4])

---

### In short

* We add a small **evals** module to `pydantic-ai-helpers`.
* You get **safe field access**, **consistent normalization**, **clean scalar/list/inclusion comparators**, and **ready‑made evaluators** that always return `EvaluationReason(value, reason)`.
* It’s thoroughly testable, composable, and serializes cleanly inside Pydantic Evals reports.

If you want, I can also generate PR‑ready files (with a `tests/` folder and a short docs page) matching your repository structure and CI preferences.

[1]: https://ai.pydantic.dev/api/pydantic_evals/evaluators/ "pydantic_evals.evaluators - Pydantic AI"
[2]: https://ai.pydantic.dev/api/pydantic_evals/evaluators/?utm_source=chatgpt.com "pydantic_evals.evaluators"
[3]: https://pypi.org/project/pydantic-evals/ "pydantic-evals · PyPI"
[4]: https://siml.earth/pydantic-ai-helpers/ "pydantic-ai-helpers | Boring, opinionated helpers for PydanticAI that are so dumb you didn’t want to implement them"
