"""Test suite for evaluator implementations."""

from dataclasses import dataclass
from unittest.mock import Mock

from pydantic_ai_helpers.evals.compare import ScalarCompare
from pydantic_ai_helpers.evals.evaluators import (
    CompareFields,
    ListEquality,
    ListPrecision,
    ListRecall,
    MaxCount,
    MaxLength,
    MinCount,
    MinLength,
    ScalarEquals,
    ValueInExpectedList,
)
from pydantic_ai_helpers.evals.normalize import (
    CompareOptions,
    FuzzyOptions,
)
from pydantic_evals.evaluators import EvaluationReason, EvaluatorContext


@dataclass
class MockContext:
    """Mock context for testing evaluators."""

    inputs: object
    output: object
    expected_output: object


def create_mock_context(inputs=None, output=None, expected_output=None):
    """Create a mock EvaluatorContext for testing."""
    context = Mock(spec=EvaluatorContext)
    context.inputs = inputs
    context.output = output
    context.expected_output = expected_output
    return context


class TestCompareFields:
    """Test the CompareFields base evaluator."""

    def test_successful_comparison(self) -> None:
        """Test successful field comparison."""
        comparator = Mock()
        comparator.return_value = (0.8, "test reason")

        evaluator = CompareFields(
            output_path="user.name",
            expected_path="user.name",
            comparator=comparator,
            evaluation_name="name_test",
        )

        ctx = create_mock_context(
            output={"user": {"name": "Alice"}},
            expected_output={"user": {"name": "Alice"}},
        )

        result = evaluator.evaluate(ctx)
        assert isinstance(result, EvaluationReason)
        assert result.value == 0.8
        assert result.reason == "[name_test] test reason"

        # Verify comparator was called with extracted values
        comparator.assert_called_once_with("Alice", "Alice")

    def test_output_path_error(self) -> None:
        """Test handling of output path errors."""
        evaluator = CompareFields(
            output_path="missing.field", expected_path="user.name", comparator=Mock()
        )

        ctx = create_mock_context(
            output={"user": {"name": "Alice"}},
            expected_output={"user": {"name": "Alice"}},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "output path error" in result.reason

    def test_expected_path_error(self) -> None:
        """Test handling of expected path errors."""
        evaluator = CompareFields(
            output_path="user.name", expected_path="missing.field", comparator=Mock()
        )

        ctx = create_mock_context(
            output={"user": {"name": "Alice"}},
            expected_output={"user": {"name": "Alice"}},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "expected path error" in result.reason

    def test_missing_expected_output(self) -> None:
        """Test handling of missing expected_output - should pass."""
        evaluator = CompareFields(
            output_path="user.name", expected_path="user.name", comparator=Mock()
        )

        ctx = create_mock_context(
            output={"user": {"name": "Alice"}}, expected_output=None
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "no expected_output, passing" in result.reason

    def test_missing_comparator(self) -> None:
        """Test handling of missing comparator."""
        evaluator = CompareFields(
            output_path="user.name", expected_path="user.name", comparator=None
        )

        ctx = create_mock_context(
            output={"user": {"name": "Alice"}},
            expected_output={"user": {"name": "Alice"}},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "no comparator provided" in result.reason

    def test_root_path_access(self) -> None:
        """Test accessing root objects (None paths)."""
        comparator = Mock()
        comparator.return_value = (1.0, "match")

        evaluator = CompareFields(
            output_path=None, expected_path=None, comparator=comparator
        )

        ctx = create_mock_context(output="test_output", expected_output="test_expected")

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0

        # Should compare the root objects directly
        comparator.assert_called_once_with("test_output", "test_expected")


class TestScalarEquals:
    """Test the ScalarEquals evaluator."""

    def test_initialization_with_options(self) -> None:
        """Test that ScalarEquals properly initializes with new options system."""
        # Test with structured options
        opts = CompareOptions(
            coerce_to="float", abs_tol=0.01, fuzzy=FuzzyOptions(enabled=False)
        )
        evaluator = ScalarEquals(
            output_path="price",
            expected_path="price",
            compare_options=opts,
            evaluation_name="price_test",
        )

        # Check that comparator was set
        assert evaluator.comparator is not None
        assert isinstance(evaluator.comparator, ScalarCompare)

    def test_initialization_with_flat_options(self) -> None:
        """Test initialization with flat options."""
        evaluator = ScalarEquals(
            output_path="price",
            expected_path="price",
            coerce_to="float",
            abs_tol=0.01,
            fuzzy_enabled=False,
            evaluation_name="price_test",
        )

        # Check that comparator was set
        assert evaluator.comparator is not None
        assert isinstance(evaluator.comparator, ScalarCompare)

    def test_numeric_comparison(self) -> None:
        """Test numeric comparison with tolerance."""
        evaluator = ScalarEquals(
            output_path="price",
            expected_path="price",
            coerce_to="float",
            abs_tol=0.01,
            fuzzy_enabled=False,  # Disable fuzzy for numeric comparison
        )

        ctx = create_mock_context(
            output={"price": "3.14"},
            expected_output={"price": 3.135},  # Within tolerance
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "numbers match" in result.reason

    def test_string_fuzzy_matching(self) -> None:
        """Test string fuzzy matching with new defaults."""
        evaluator = ScalarEquals(
            output_path="name",
            expected_path="name",
            normalize_lowercase=True,
            normalize_strip=True,
        )

        ctx = create_mock_context(
            output={"name": "  ALICE  "}, expected_output={"name": "alice"}
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "fuzzy match" in result.reason  # Should use fuzzy matching

    def test_string_exact_matching(self) -> None:
        """Test exact string matching when fuzzy is disabled."""
        evaluator = ScalarEquals(
            output_path="name",
            expected_path="name",
            normalize_lowercase=True,
            normalize_strip=True,
            fuzzy_enabled=False,
        )

        ctx = create_mock_context(
            output={"name": "  ALICE  "}, expected_output={"name": "alice"}
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "values equal" in result.reason


class TestListEvaluators:
    """Test list-based evaluators."""

    def test_list_equality_with_fuzzy(self) -> None:
        """Test ListEquality evaluator with fuzzy matching."""
        evaluator = ListEquality(
            output_path="tags",
            expected_path="tags",
            normalize_lowercase=True,
            fuzzy_enabled=True,
        )

        ctx = create_mock_context(
            output={"tags": ["Python", "AI"]},
            expected_output={"tags": ["python", "ai"]},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "fuzzy equality" in result.reason

    def test_list_equality_exact(self) -> None:
        """Test ListEquality with exact matching."""
        evaluator = ListEquality(
            output_path="tags",
            expected_path="tags",
            normalize_lowercase=True,
            fuzzy_enabled=False,
        )

        ctx = create_mock_context(
            output={"tags": ["Python", "AI"]},
            expected_output={"tags": ["python", "ai"]},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "lists equal" in result.reason

    def test_list_recall_with_fuzzy(self) -> None:
        """Test ListRecall evaluator with fuzzy matching."""
        evaluator = ListRecall(
            output_path="predicted", expected_path="required", fuzzy_enabled=True
        )

        ctx = create_mock_context(
            output={"predicted": ["a", "b"]},
            expected_output={"required": ["a", "b", "c"]},
        )

        result = evaluator.evaluate(ctx)
        # With fuzzy matching, exact matches should still work
        assert result.value >= 2 / 3  # At least as good as exact
        assert "fuzzy recall" in result.reason

    def test_list_precision_with_fuzzy(self) -> None:
        """Test ListPrecision evaluator with fuzzy matching."""
        evaluator = ListPrecision(
            output_path="predicted", expected_path="valid", fuzzy_enabled=True
        )

        ctx = create_mock_context(
            output={"predicted": ["a", "b", "x"]},
            expected_output={"valid": ["a", "b", "c"]},
        )

        result = evaluator.evaluate(ctx)
        # With fuzzy matching, should get at least exact match score
        assert result.value >= 2 / 3
        assert "fuzzy precision" in result.reason


class TestValueInExpectedList:
    """Test the ValueInExpectedList evaluator."""

    def test_value_in_list_with_fuzzy(self) -> None:
        """Test successful inclusion check with fuzzy matching."""
        evaluator = ValueInExpectedList(
            output_path="category",
            expected_path="valid_categories",
            normalize_lowercase=True,
            fuzzy_enabled=True,
        )

        ctx = create_mock_context(
            output={"category": "SCIENCE"},
            expected_output={"valid_categories": ["science", "math", "history"]},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "fuzzy match" in result.reason

    def test_value_in_list_exact(self) -> None:
        """Test inclusion check with exact matching."""
        evaluator = ValueInExpectedList(
            output_path="category",
            expected_path="valid_categories",
            normalize_lowercase=True,
            fuzzy_enabled=False,
        )

        ctx = create_mock_context(
            output={"category": "SCIENCE"},
            expected_output={"valid_categories": ["science", "math", "history"]},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "'science' in" in result.reason

    def test_value_not_in_list_fuzzy(self) -> None:
        """Test failed inclusion check with fuzzy matching."""
        evaluator = ValueInExpectedList(
            output_path="category", expected_path="valid_categories", fuzzy_enabled=True
        )

        ctx = create_mock_context(
            output={"category": "invalid"},
            expected_output={"valid_categories": ["science", "math", "history"]},
        )

        result = evaluator.evaluate(ctx)
        # Should get some fuzzy score but below threshold
        assert 0.0 <= result.value < 0.85
        assert "no fuzzy match" in result.reason

    def test_value_not_in_list_exact(self) -> None:
        """Test failed inclusion check with exact matching."""
        evaluator = ValueInExpectedList(
            output_path="category",
            expected_path="valid_categories",
            fuzzy_enabled=False,
        )

        ctx = create_mock_context(
            output={"category": "invalid"},
            expected_output={"valid_categories": ["science", "math", "history"]},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "'invalid' not in" in result.reason


# MultiCompare tests removed as the feature has been dropped.
# Keeping the integration tests below


class TestEvaluatorIntegration:
    """Test evaluators with realistic scenarios."""

    def test_real_scenario_scalar(self) -> None:
        """Test a realistic scalar evaluation scenario."""
        evaluator = ScalarEquals(
            output_path="user.age",
            expected_path="user.age",
            coerce_to="int",
            fuzzy_enabled=False,  # Disable fuzzy for numeric comparison
            evaluation_name="age_check",
        )

        ctx = create_mock_context(
            output={"user": {"name": "Alice", "age": "25"}},
            expected_output={"user": {"name": "Alice", "age": 25}},
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "[age_check]" in result.reason

    def test_real_scenario_list_with_fuzzy(self) -> None:
        """Test a realistic list evaluation scenario with fuzzy matching."""
        evaluator = ListRecall(
            output_path="predicted_tags",
            expected_path="required_tags",
            normalize_lowercase=True,
            normalize_strip=True,
            fuzzy_enabled=True,
            evaluation_name="tag_recall",
        )

        ctx = create_mock_context(
            output={"predicted_tags": ["  Python  ", "AI", "Machine Learning"]},
            expected_output={"required_tags": ["python", "ai", "data science", "ml"]},
        )

        result = evaluator.evaluate(ctx)
        # With fuzzy matching, should find good matches for python and ai
        # Machine Learning might partially match "ml"
        assert result.value >= 0.5  # At least as good as exact matching
        assert "[tag_recall]" in result.reason
        assert "fuzzy recall" in result.reason

    def test_nested_data_structures(self) -> None:
        """Test evaluators with complex nested data."""
        evaluator = ScalarEquals(
            output_path="results.metrics.accuracy",
            expected_path="expected.metrics.accuracy",
            coerce_to="float",
            abs_tol=0.01,
        )

        ctx = create_mock_context(
            output={"results": {"metrics": {"accuracy": "0.94", "precision": "0.88"}}},
            expected_output={
                "expected": {"metrics": {"accuracy": 0.94, "precision": 0.87}}
            },
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0  # Within tolerance


class TestMaxCount:
    """Test the MaxCount evaluator."""

    def test_value_below_max_count(self) -> None:
        """Test successful evaluation when value is below max count."""
        evaluator = MaxCount(
            output_path="item_count", count=10, evaluation_name="max_items"
        )

        ctx = create_mock_context(output={"item_count": 5})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "value 5 is <= max count 10" in result.reason
        assert "[max_items]" in result.reason

    def test_value_equal_to_max_count(self) -> None:
        """Test successful evaluation when value equals max count."""
        evaluator = MaxCount(output_path="item_count", count=10)

        ctx = create_mock_context(output={"item_count": 10})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "value 10 is <= max count 10" in result.reason

    def test_value_above_max_count(self) -> None:
        """Test failed evaluation when value exceeds max count."""
        evaluator = MaxCount(
            output_path="item_count", count=10, evaluation_name="max_items"
        )

        ctx = create_mock_context(output={"item_count": 15})

        result = evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "value 15 exceeds max count 10" in result.reason
        assert "[max_items]" in result.reason

    def test_string_number_conversion(self) -> None:
        """Test conversion from string to int."""
        evaluator = MaxCount(output_path="count", count=5)

        ctx = create_mock_context(output={"count": "3"})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "value 3 is <= max count 5" in result.reason

    def test_float_number_conversion(self) -> None:
        """Test conversion from float to int."""
        evaluator = MaxCount(output_path="count", count=5)

        ctx = create_mock_context(output={"count": 3.7})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "value 3 is <= max count 5" in result.reason

    def test_invalid_conversion(self) -> None:
        """Test handling of values that cannot be converted to int."""
        evaluator = MaxCount(output_path="count", count=5, evaluation_name="max_items")

        ctx = create_mock_context(output={"count": "not_a_number"})

        result = evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "cannot convert 'not_a_number' to int" in result.reason
        assert "[max_items]" in result.reason

    def test_missing_output_path(self) -> None:
        """Test handling of missing output path."""
        evaluator = MaxCount(output_path="missing.field", count=5)

        ctx = create_mock_context(output={"other_field": 3})

        result = evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "output path error" in result.reason

    def test_root_path_access(self) -> None:
        """Test accessing root object (None path)."""
        evaluator = MaxCount(output_path=None, count=10)

        ctx = create_mock_context(output=5)

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "value 5 is <= max count 10" in result.reason

    def test_missing_expected_output(self) -> None:
        """Test that missing expected_output doesn't affect constraint evaluators."""
        evaluator = MaxCount(output_path="count", count=10, evaluation_name="max_items")

        ctx = create_mock_context(
            output={"count": 5},
            expected_output=None,  # No expected output
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "value 5 is <= max count 10" in result.reason
        assert "[max_items]" in result.reason


class TestMinCount:
    """Test the MinCount evaluator."""

    def test_value_above_min_count(self) -> None:
        """Test successful evaluation when value is above min count."""
        evaluator = MinCount(
            output_path="item_count", count=5, evaluation_name="min_items"
        )

        ctx = create_mock_context(output={"item_count": 10})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "value 10 is >= min count 5" in result.reason
        assert "[min_items]" in result.reason

    def test_value_equal_to_min_count(self) -> None:
        """Test successful evaluation when value equals min count."""
        evaluator = MinCount(output_path="item_count", count=5)

        ctx = create_mock_context(output={"item_count": 5})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "value 5 is >= min count 5" in result.reason

    def test_value_below_min_count(self) -> None:
        """Test failed evaluation when value is below min count."""
        evaluator = MinCount(
            output_path="item_count", count=5, evaluation_name="min_items"
        )

        ctx = create_mock_context(output={"item_count": 2})

        result = evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "value 2 is below min count 5" in result.reason
        assert "[min_items]" in result.reason

    def test_negative_numbers(self) -> None:
        """Test handling of negative numbers."""
        evaluator = MinCount(output_path="count", count=-5)

        ctx = create_mock_context(output={"count": -3})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "value -3 is >= min count -5" in result.reason

    def test_zero_count(self) -> None:
        """Test handling of zero values."""
        evaluator = MinCount(output_path="count", count=0)

        ctx = create_mock_context(output={"count": 0})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "value 0 is >= min count 0" in result.reason

    def test_invalid_conversion(self) -> None:
        """Test handling of values that cannot be converted to int."""
        evaluator = MinCount(output_path="count", count=5, evaluation_name="min_items")

        ctx = create_mock_context(output={"count": None})

        result = evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "cannot convert 'None' to int" in result.reason
        assert "[min_items]" in result.reason

    def test_missing_expected_output(self) -> None:
        """Test that missing expected_output doesn't affect constraint evaluators."""
        evaluator = MinCount(output_path="count", count=3, evaluation_name="min_items")

        ctx = create_mock_context(
            output={"count": 5},
            expected_output=None,  # No expected output
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "value 5 is >= min count 3" in result.reason
        assert "[min_items]" in result.reason


class TestMaxLength:
    """Test the MaxLength evaluator."""

    def test_string_below_max_length(self) -> None:
        """Test successful evaluation when string is below max length."""
        evaluator = MaxLength(
            output_path="description", length=20, evaluation_name="max_desc_length"
        )

        ctx = create_mock_context(output={"description": "Short text"})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 10 is <= max length 20" in result.reason
        assert "[max_desc_length]" in result.reason

    def test_string_equal_to_max_length(self) -> None:
        """Test successful evaluation when string equals max length."""
        evaluator = MaxLength(output_path="description", length=10)

        ctx = create_mock_context(output={"description": "Exactly 10"})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 10 is <= max length 10" in result.reason

    def test_string_above_max_length(self) -> None:
        """Test failed evaluation when string exceeds max length."""
        evaluator = MaxLength(
            output_path="description", length=10, evaluation_name="max_desc_length"
        )

        ctx = create_mock_context(
            output={"description": "This is a very long description"}
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "string length 31 exceeds max length 10" in result.reason
        assert "[max_desc_length]" in result.reason

    def test_string_stripping(self) -> None:
        """Test that strings are properly stripped before length check."""
        evaluator = MaxLength(output_path="text", length=5)

        ctx = create_mock_context(output={"text": "  hello  "})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 5 is <= max length 5" in result.reason

    def test_empty_string(self) -> None:
        """Test handling of empty strings."""
        evaluator = MaxLength(output_path="text", length=5)

        ctx = create_mock_context(output={"text": ""})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 0 is <= max length 5" in result.reason

    def test_whitespace_only_string(self) -> None:
        """Test handling of whitespace-only strings."""
        evaluator = MaxLength(output_path="text", length=5)

        ctx = create_mock_context(output={"text": "   "})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 0 is <= max length 5" in result.reason

    def test_non_string_conversion(self) -> None:
        """Test conversion of non-string values to string."""
        evaluator = MaxLength(output_path="value", length=5)

        ctx = create_mock_context(output={"value": 123})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 3 is <= max length 5" in result.reason

    def test_zero_max_length(self) -> None:
        """Test handling of zero max length."""
        evaluator = MaxLength(output_path="text", length=0)

        ctx = create_mock_context(output={"text": ""})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 0 is <= max length 0" in result.reason

    def test_missing_output_path(self) -> None:
        """Test handling of missing output path."""
        evaluator = MaxLength(output_path="missing.field", length=10)

        ctx = create_mock_context(output={"other_field": "text"})

        result = evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "output path error" in result.reason

    def test_missing_expected_output(self) -> None:
        """Test that missing expected_output doesn't affect constraint evaluators."""
        evaluator = MaxLength(
            output_path="description", length=20, evaluation_name="max_desc_length"
        )

        ctx = create_mock_context(
            output={"description": "Short text"},
            expected_output=None,  # No expected output
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 10 is <= max length 20" in result.reason
        assert "[max_desc_length]" in result.reason


class TestMinLength:
    """Test the MinLength evaluator."""

    def test_string_above_min_length(self) -> None:
        """Test successful evaluation when string is above min length."""
        evaluator = MinLength(
            output_path="description", length=5, evaluation_name="min_desc_length"
        )

        ctx = create_mock_context(output={"description": "This is a long description"})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 26 is >= min length 5" in result.reason
        assert "[min_desc_length]" in result.reason

    def test_string_equal_to_min_length(self) -> None:
        """Test successful evaluation when string equals min length."""
        evaluator = MinLength(output_path="description", length=5)

        ctx = create_mock_context(output={"description": "hello"})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 5 is >= min length 5" in result.reason

    def test_string_below_min_length(self) -> None:
        """Test failed evaluation when string is below min length."""
        evaluator = MinLength(
            output_path="description", length=10, evaluation_name="min_desc_length"
        )

        ctx = create_mock_context(output={"description": "short"})

        result = evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "string length 5 is below min length 10" in result.reason
        assert "[min_desc_length]" in result.reason

    def test_empty_string_with_min_length(self) -> None:
        """Test empty string against min length requirement."""
        evaluator = MinLength(output_path="text", length=1)

        ctx = create_mock_context(output={"text": ""})

        result = evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "string length 0 is below min length 1" in result.reason

    def test_zero_min_length(self) -> None:
        """Test handling of zero min length."""
        evaluator = MinLength(output_path="text", length=0)

        ctx = create_mock_context(output={"text": ""})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 0 is >= min length 0" in result.reason

    def test_unicode_string_length(self) -> None:
        """Test handling of unicode strings."""
        evaluator = MinLength(output_path="text", length=3)

        ctx = create_mock_context(output={"text": "🚀💫✨"})

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 3 is >= min length 3" in result.reason

    def test_list_conversion_to_string(self) -> None:
        """Test conversion of list to string for length check."""
        evaluator = MinLength(output_path="items", length=10)

        ctx = create_mock_context(output={"items": [1, 2, 3]})

        result = evaluator.evaluate(ctx)
        # List converts to "[1, 2, 3]" which is 9 characters
        assert result.value == 0.0
        assert "string length 9 is below min length 10" in result.reason

    def test_root_path_access(self) -> None:
        """Test accessing root object (None path)."""
        evaluator = MinLength(output_path=None, length=5)

        ctx = create_mock_context(output="hello world")

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 11 is >= min length 5" in result.reason

    def test_missing_expected_output(self) -> None:
        """Test that missing expected_output doesn't affect constraint evaluators."""
        evaluator = MinLength(
            output_path="description", length=5, evaluation_name="min_desc_length"
        )

        ctx = create_mock_context(
            output={"description": "This is a long description"},
            expected_output=None,  # No expected output
        )

        result = evaluator.evaluate(ctx)
        assert result.value == 1.0
        assert "string length 26 is >= min length 5" in result.reason
        assert "[min_desc_length]" in result.reason


class TestCountAndLengthEvaluatorIntegration:
    """Test integration scenarios with count and length evaluators."""

    def test_realistic_content_validation_scenario(self) -> None:
        """Test a realistic content validation scenario."""
        # Validate blog post has appropriate title length and tag count
        min_title_evaluator = MinLength(
            output_path="post.title", length=10, evaluation_name="min_title_length"
        )

        max_title_evaluator = MaxLength(
            output_path="post.title", length=100, evaluation_name="max_title_length"
        )

        min_tags_evaluator = MinCount(
            output_path="post.tag_count", count=2, evaluation_name="min_tags"
        )

        max_tags_evaluator = MaxCount(
            output_path="post.tag_count", count=10, evaluation_name="max_tags"
        )

        ctx = create_mock_context(
            output={
                "post": {
                    "title": "Understanding Machine Learning Fundamentals",
                    "tag_count": 5,
                }
            }
        )

        # All validations should pass
        results = [
            min_title_evaluator.evaluate(ctx),
            max_title_evaluator.evaluate(ctx),
            min_tags_evaluator.evaluate(ctx),
            max_tags_evaluator.evaluate(ctx),
        ]

        for result in results:
            assert result.value == 1.0

    def test_validation_failures(self) -> None:
        """Test scenarios where validation should fail."""
        evaluators = [
            MaxCount(output_path="items", count=5, evaluation_name="item_limit"),
            MinLength(output_path="description", length=20, evaluation_name="desc_min"),
        ]

        ctx = create_mock_context(
            output={
                "items": 10,  # Exceeds max count of 5
                "description": "Too short",  # Below min length of 20
            }
        )

        results = [evaluator.evaluate(ctx) for evaluator in evaluators]

        # Both should fail
        assert all(result.value == 0.0 for result in results)
        assert "exceeds max count 5" in results[0].reason
        assert "below min length 20" in results[1].reason
