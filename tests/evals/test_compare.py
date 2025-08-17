"""Test suite for comparison utilities."""

from enum import Enum

import pytest

from pydantic_ai_helpers.evals.compare import (
    CoercionError,
    InclusionCompare,
    ListCompare,
    ScalarCompare,
    _coerce,
    _coerce_bool,
)
from pydantic_ai_helpers.evals.normalize import (
    CompareOptions,
    FuzzyOptions,
)


class Color(Enum):
    """Sample enum for testing."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class TestCoercionUtils:
    """Test type coercion utilities."""

    def test_coerce_bool(self) -> None:
        """Test boolean coercion."""
        # True values
        assert _coerce_bool(True) is True
        assert _coerce_bool(1) is True
        assert _coerce_bool(42) is True
        assert _coerce_bool("true") is True
        assert _coerce_bool("TRUE") is True
        assert _coerce_bool("  True  ") is True
        assert _coerce_bool("t") is True
        assert _coerce_bool("yes") is True
        assert _coerce_bool("y") is True
        assert _coerce_bool("1") is True

        # False values
        assert _coerce_bool(False) is False
        assert _coerce_bool(0) is False
        assert _coerce_bool(0.0) is False
        assert _coerce_bool("false") is False
        assert _coerce_bool("FALSE") is False
        assert _coerce_bool("  False  ") is False
        assert _coerce_bool("f") is False
        assert _coerce_bool("no") is False
        assert _coerce_bool("n") is False
        assert _coerce_bool("0") is False

        # Invalid values
        with pytest.raises(CoercionError):
            _coerce_bool("maybe")
        with pytest.raises(CoercionError):
            _coerce_bool("2")
        with pytest.raises(CoercionError):
            _coerce_bool([])

    def test_coerce_basic_types(self) -> None:
        """Test coercion to basic types."""
        # String coercion
        assert _coerce(123, "str") == "123"
        assert _coerce(3.14, "str") == "3.14"
        assert _coerce(True, "str") == "True"

        # Integer coercion
        assert _coerce("123", "int") == 123
        assert _coerce(3.14, "int") == 3
        assert _coerce(True, "int") == 1

        # Float coercion
        assert _coerce("3.14", "float") == 3.14
        assert _coerce(42, "float") == 42.0
        assert _coerce(True, "float") == 1.0

        # Boolean coercion
        assert _coerce("true", "bool") is True
        assert _coerce("false", "bool") is False
        assert _coerce(1, "bool") is True
        assert _coerce(0, "bool") is False

    def test_coerce_enum(self) -> None:
        """Test enum coercion."""
        # With enum values
        assert _coerce("red", "enum", enum_values=["red", "green", "blue"]) == "red"
        assert _coerce("RED", "enum", enum_values=["red", "green", "blue"]) == "red"
        assert _coerce("  Red  ", "enum", enum_values=["red", "green", "blue"]) == "red"

        # With enum class (returns member names, which are uppercase)
        assert _coerce("red", Color) == "RED"
        assert _coerce("GREEN", Color) == "GREEN"

        # Invalid enum value
        with pytest.raises(CoercionError):
            _coerce("purple", "enum", enum_values=["red", "green", "blue"])

        # Missing enum values
        with pytest.raises(CoercionError):
            _coerce("red", "enum")

    def test_coerce_errors(self) -> None:
        """Test coercion error cases."""
        # Invalid target type
        with pytest.raises(CoercionError):
            _coerce("test", "invalid_type")

        # Invalid conversions
        with pytest.raises(ValueError):  # from int() conversion
            _coerce("not_a_number", "int")

        with pytest.raises(ValueError):  # from float() conversion
            _coerce("not_a_float", "float")


class TestScalarCompare:
    """Test scalar comparison functionality."""

    def test_basic_equality_with_fuzzy_defaults(self) -> None:
        """Test basic equality with new fuzzy defaults."""
        comp = ScalarCompare()

        # Exact matches should still work
        value, reason = comp("hello", "hello")
        assert value == 1.0
        assert "fuzzy match" in reason

        # Different strings should get fuzzy scores
        value, reason = comp("hello", "world")
        assert 0.0 <= value < 0.85  # Below default threshold
        assert "fuzzy no match" in reason

    def test_fuzzy_disabled(self) -> None:
        """Test comparison with fuzzy matching disabled."""
        opts = CompareOptions(fuzzy=FuzzyOptions(enabled=False))
        comp = ScalarCompare(options=opts)

        value, reason = comp("hello", "hello")
        assert value == 1.0
        assert "values equal" in reason

        value, reason = comp("hello", "world")
        assert value == 0.0
        assert "values differ" in reason

    def test_fuzzy_with_normalization(self) -> None:
        """Test fuzzy matching with normalization."""
        comp = ScalarCompare()  # Uses defaults

        # Should match after normalization
        value, reason = comp("  HELLO  ", "hello")
        assert value == 1.0
        assert "fuzzy match" in reason

        # Close match should get partial score
        value, reason = comp("hello", "helo")
        assert 0.8 < value < 1.0
        assert "fuzzy match" in reason or "fuzzy no match" in reason

    def test_custom_fuzzy_threshold(self) -> None:
        """Test custom fuzzy threshold."""
        opts = CompareOptions(fuzzy=FuzzyOptions(threshold=0.95))
        comp = ScalarCompare(options=opts)

        # Close match should fail with high threshold
        value, reason = comp("hello", "helo")
        assert value < 0.95
        assert "fuzzy no match" in reason

    def test_numeric_comparison(self) -> None:
        """Test numeric comparisons with tolerance."""
        # Exact match
        opts = CompareOptions(abs_tol=0.0, rel_tol=0.0)
        comp = ScalarCompare(options=opts)
        value, reason = comp(3.14, 3.14)
        assert value == 1.0
        assert "numbers match" in reason

        # With absolute tolerance
        opts = CompareOptions(abs_tol=0.1)
        comp = ScalarCompare(options=opts)
        value, reason = comp(3.14, 3.1)
        assert value == 1.0
        assert "numbers match" in reason

        value, reason = comp(3.14, 3.0)
        assert value == 0.0
        assert "numbers differ" in reason

        # With relative tolerance
        opts = CompareOptions(rel_tol=0.01)
        comp = ScalarCompare(options=opts)
        value, reason = comp(100.0, 100.5)
        assert value == 1.0
        assert "numbers match" in reason

    def test_type_coercion(self) -> None:
        """Test comparisons with type coercion."""
        # String to float
        comp = ScalarCompare(coerce_to="float")
        value, reason = comp("3.14", 3.14)
        assert value == 1.0
        assert "numbers match" in reason

        # String to int
        comp = ScalarCompare(coerce_to="int")
        value, reason = comp("42", 42)
        assert value == 1.0
        assert "numbers match" in reason

        # Boolean coercion
        comp = ScalarCompare(coerce_to="bool")
        value, reason = comp("true", True)
        assert value == 1.0
        assert "numbers match" in reason

    def test_normalization(self) -> None:
        """Test string normalization."""
        comp = ScalarCompare(normalize_opts={"lowercase": True, "strip": True})
        value, reason = comp("  HELLO  ", "hello")
        assert value == 1.0
        assert "values equal" in reason

        comp = ScalarCompare(normalize_opts={"alphanum": True})
        value, reason = comp("hello-world", "helloworld")
        assert value == 1.0
        assert "values equal" in reason

    def test_enum_comparison(self) -> None:
        """Test enum comparisons."""
        comp = ScalarCompare(coerce_to="enum", enum_values=["red", "green", "blue"])
        value, reason = comp("red", "RED")
        assert value == 1.0
        assert "values equal" in reason

        comp = ScalarCompare(coerce_to=Color)
        value, reason = comp("red", "RED")
        assert value == 1.0
        assert "values equal" in reason

    def test_coercion_failures(self) -> None:
        """Test handling of coercion failures."""
        comp = ScalarCompare(coerce_to="float")

        value, reason = comp("not_a_number", 3.14)
        assert value == 0.0
        assert "left coercion failed" in reason

        value, reason = comp(3.14, "not_a_number")
        assert value == 0.0
        assert "right coercion failed" in reason

    def test_non_finite_numbers(self) -> None:
        """Test handling of non-finite numbers."""
        comp = ScalarCompare()

        value, reason = comp(float("inf"), 3.14)
        assert value == 0.0
        assert "non-finite number" in reason

        value, reason = comp(3.14, float("nan"))
        assert value == 0.0
        assert "non-finite number" in reason


class TestListCompare:
    """Test list comparison functionality."""

    def test_equality_mode(self) -> None:
        """Test list equality comparisons."""
        comp = ListCompare(mode="equality")

        # Order insensitive (default)
        value, reason = comp(["a", "b"], ["b", "a"])
        assert value == 1.0
        assert "fuzzy equality" in reason

        # Order sensitive
        comp = ListCompare(mode="equality", order_sensitive=True)
        value, reason = comp(["a", "b"], ["b", "a"])
        assert value == 0.0
        assert "fuzzy equality" in reason and "avg_score=0.000" in reason

        value, reason = comp(["a", "b"], ["a", "b"])
        assert value == 1.0
        assert "fuzzy equality" in reason

    def test_multiset_equality(self) -> None:
        """Test multiset equality comparisons."""
        comp = ListCompare(mode="equality", multiset=True)

        value, reason = comp(["a", "a", "b"], ["a", "b", "a"])
        assert value == 1.0
        assert "fuzzy equality" in reason

        value, reason = comp(["a", "a", "b"], ["a", "b"])
        assert value == 0.0
        assert "lists have different lengths" in reason

    def test_recall_mode(self) -> None:
        """Test recall calculations."""
        comp = ListCompare(mode="recall")

        # Perfect recall
        value, reason = comp(["a", "b", "c"], ["a", "b", "c"])
        assert value == 1.0
        assert "fuzzy recall: fuzzy_score_sum=3.000, denom=3, score=1.0000" in reason

        # Partial recall
        value, reason = comp(["a", "b"], ["a", "b", "c"])
        assert abs(value - 2 / 3) < 0.001
        assert "fuzzy recall: fuzzy_score_sum=2.000, denom=3" in reason

        # No requirements
        value, reason = comp(["a", "b"], [])
        assert value == 1.0
        assert "denominator=0" in reason

    def test_precision_mode(self) -> None:
        """Test precision calculations."""
        comp = ListCompare(mode="precision")

        # Perfect precision
        value, reason = comp(["a", "b"], ["a", "b", "c"])
        assert value == 1.0
        assert "fuzzy precision: fuzzy_score_sum=2.000, denom=2, score=1.0000" in reason

        # Partial precision
        value, reason = comp(["a", "b", "x"], ["a", "b", "c"])
        assert abs(value - 2 / 3) < 0.001
        assert "fuzzy precision: fuzzy_score_sum=2.000, denom=3" in reason

        # No output
        value, reason = comp([], ["a", "b"])
        assert value == 1.0
        assert "denominator=0" in reason

    def test_normalization(self) -> None:
        """Test element normalization."""
        comp = ListCompare(
            mode="equality", normalize_opts={"lowercase": True, "strip": True}
        )

        value, reason = comp(["  HELLO  ", "WORLD"], ["hello", "world"])
        assert value == 1.0
        assert "lists equal" in reason

    def test_element_coercion(self) -> None:
        """Test element type coercion."""
        comp = ListCompare(mode="equality", element_coerce_to="int")

        value, reason = comp(["1", "2", "3"], [1, 2, 3])
        assert value == 1.0
        assert "lists equal" in reason

        # Partial coercion (some elements fail)
        value, reason = comp(["1", "not_a_number", "3"], [1, 2, 3])
        assert value == 0.0
        assert "lists differ" in reason

    def test_multiset_recall_precision(self) -> None:
        """Test multiset recall and precision."""
        # Multiset recall
        comp = ListCompare(mode="recall", multiset=True)
        value, reason = comp(["a", "a", "b"], ["a", "a", "a", "b", "b"])
        # hits: a=2 (min of 2,3), b=1 (min of 1,2) = 3 total
        # denom: 5 (total in expected)
        assert abs(value - 3 / 5) < 0.001

        # Multiset precision
        comp = ListCompare(mode="precision", multiset=True)
        value, reason = comp(["a", "a", "b"], ["a", "a", "a", "b", "b"])
        # hits: same 3 as above
        # denom: 3 (total in output)
        assert value == 1.0

    def test_invalid_input_types(self) -> None:
        """Test error handling for invalid input types."""
        comp = ListCompare()

        # String input (not a sequence for our purposes)
        value, reason = comp("hello", ["h", "e", "l", "l", "o"])
        assert value == 0.0
        assert "left is not a sequence" in reason

        # Non-iterable input
        value, reason = comp([1, 2, 3], 42)
        assert value == 0.0
        assert "right is not a sequence" in reason


class TestInclusionCompare:
    """Test inclusion comparison functionality."""

    def test_basic_inclusion(self) -> None:
        """Test basic value inclusion checks."""
        comp = InclusionCompare()

        value, reason = comp("apple", ["apple", "banana", "cherry"])
        assert value == 1.0
        assert "fuzzy match: 'apple' -> 'apple'" in reason

        value, reason = comp("grape", ["apple", "banana", "cherry"])
        assert value < 0.85  # Below fuzzy threshold
        assert "no fuzzy match" in reason or "best_score" in reason

    def test_normalization(self) -> None:
        """Test normalization in inclusion checks."""
        comp = InclusionCompare(normalize_opts={"lowercase": True, "strip": True})

        value, reason = comp("  APPLE  ", ["apple", "banana"])
        assert value == 1.0
        assert "'apple' in" in reason

    def test_element_coercion(self) -> None:
        """Test element coercion in inclusion checks."""
        comp = InclusionCompare(element_coerce_to="int")

        value, reason = comp("42", [1, 42, 100])
        assert value == 1.0
        assert "42 in" in reason

        value, reason = comp("99", [1, 42, 100])
        assert value == 0.0
        assert "99 not in" in reason

    def test_coercion_failure(self) -> None:
        """Test handling of coercion failures."""
        comp = InclusionCompare(element_coerce_to="int")

        value, reason = comp("not_a_number", [1, 2, 3])
        assert value == 0.0
        assert "left coercion failed" in reason

    def test_invalid_sequence_type(self) -> None:
        """Test error handling for invalid sequence types."""
        comp = InclusionCompare()

        # String as sequence (should be treated as non-sequence)
        value, reason = comp("a", "abc")
        assert value == 0.0
        assert "right is not a sequence" in reason

        # Non-iterable
        value, reason = comp("test", 42)
        assert value == 0.0
        assert "right is not a sequence" in reason

    def test_empty_sequence(self) -> None:
        """Test inclusion check with empty sequence."""
        comp = InclusionCompare()

        value, reason = comp("test", [])
        assert value == 0.0
        assert "no fuzzy match: 'test' best_score=0.000" in reason

    def test_mixed_types_in_sequence(self) -> None:
        """Test inclusion with mixed types in sequence."""
        comp = InclusionCompare()

        value, reason = comp("42", ["hello", 42, "world"])
        assert value == 0.0  # String "42" != int 42
        assert "no fuzzy match: '42' best_score=0.000" in reason

        value, reason = comp(42, ["hello", 42, "world"])
        assert value == 1.0
        assert "42 in" in reason
