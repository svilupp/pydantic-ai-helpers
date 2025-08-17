"""Test suite for registry utilities."""

from unittest.mock import Mock

import pytest

from pydantic_ai_helpers.evals.evaluators import (
    CompareFields,
    ListRecall,
    ScalarEquals,
    ValueInExpectedList,
)
from pydantic_ai_helpers.evals.registry import from_specs, register


class TestRegister:
    """Test the register function."""

    def test_register_single_evaluator(self) -> None:
        """Test registering a single evaluator."""
        dataset = Mock()
        evaluator = ScalarEquals(output_path="test", expected_path="test")

        register(dataset, evaluator)

        dataset.add_evaluator.assert_called_once_with(evaluator)

    def test_register_multiple_evaluators(self) -> None:
        """Test registering multiple evaluators."""
        dataset = Mock()
        eval1 = ScalarEquals(output_path="test1", expected_path="test1")
        eval2 = ListRecall(output_path="test2", expected_path="test2")
        eval3 = ValueInExpectedList(output_path="test3", expected_path="test3")

        register(dataset, eval1, eval2, eval3)

        assert dataset.add_evaluator.call_count == 3
        dataset.add_evaluator.assert_any_call(eval1)
        dataset.add_evaluator.assert_any_call(eval2)
        dataset.add_evaluator.assert_any_call(eval3)

    def test_register_no_evaluators(self) -> None:
        """Test registering with no evaluators."""
        dataset = Mock()

        register(dataset)

        dataset.add_evaluator.assert_not_called()


class TestFromSpecs:
    """Test the from_specs function."""

    def test_scalar_equals_spec(self) -> None:
        """Test creating ScalarEquals from spec."""
        dataset = Mock()
        specs = [
            {
                "kind": "ScalarEquals",
                "output_path": "price",
                "expected_path": "price",
                "coerce_to": "float",
                "abs_tol": 0.01,
                "name": "price_check",
            }
        ]

        from_specs(dataset, specs)

        dataset.add_evaluator.assert_called_once()
        evaluator = dataset.add_evaluator.call_args[0][0]

        assert isinstance(evaluator, ScalarEquals)
        assert evaluator.output_path == "price"
        assert evaluator.expected_path == "price"
        assert evaluator.coerce_to == "float"
        assert evaluator.abs_tol == 0.01
        assert evaluator.evaluation_name == "price_check"

    def test_list_recall_spec(self) -> None:
        """Test creating ListRecall from spec."""
        dataset = Mock()
        specs = [
            {
                "kind": "ListRecall",
                "output_path": "tags",
                "expected_path": "required_tags",
                "normalize": {"lowercase": True, "strip": True},
                "name": "tag_recall",
            }
        ]

        from_specs(dataset, specs)

        dataset.add_evaluator.assert_called_once()
        evaluator = dataset.add_evaluator.call_args[0][0]

        assert isinstance(evaluator, ListRecall)
        assert evaluator.output_path == "tags"
        assert evaluator.expected_path == "required_tags"
        # Check that the new structured interface is used with flat convenience params
        assert evaluator.normalize_lowercase
        assert evaluator.normalize_strip
        assert evaluator.evaluation_name == "tag_recall"

    def test_value_in_expected_list_spec(self) -> None:
        """Test creating ValueInExpectedList from spec."""
        dataset = Mock()
        specs = [
            {
                "kind": "ValueInExpectedList",
                "output_path": "category",
                "expected_path": "valid_categories",
                "element_coerce_to": "str",
            }
        ]

        from_specs(dataset, specs)

        dataset.add_evaluator.assert_called_once()
        evaluator = dataset.add_evaluator.call_args[0][0]

        assert isinstance(evaluator, ValueInExpectedList)
        assert evaluator.output_path == "category"
        assert evaluator.expected_path == "valid_categories"
        assert evaluator.element_coerce_to == "str"

    def test_compare_fields_spec(self) -> None:
        """Test creating CompareFields from spec."""
        dataset = Mock()
        specs = [
            {
                "kind": "CompareFields",
                "output_path": "custom.field",
                "expected_path": "custom.field",
                "name": "custom_comparison",
            }
        ]

        from_specs(dataset, specs)

        dataset.add_evaluator.assert_called_once()
        evaluator = dataset.add_evaluator.call_args[0][0]

        assert isinstance(evaluator, CompareFields)
        assert evaluator.output_path == "custom.field"
        assert evaluator.expected_path == "custom.field"
        assert evaluator.evaluation_name == "custom_comparison"

    def test_multiple_specs(self) -> None:
        """Test creating multiple evaluators from specs."""
        dataset = Mock()
        specs = [
            {
                "kind": "ScalarEquals",
                "output_path": "price",
                "expected_path": "price",
                "coerce_to": "float",
            },
            {
                "kind": "ListRecall",
                "output_path": "tags",
                "expected_path": "tags",
                "normalize": {"lowercase": True},
            },
            {
                "kind": "ValueInExpectedList",
                "output_path": "category",
                "expected_path": "valid_categories",
            },
        ]

        from_specs(dataset, specs)

        assert dataset.add_evaluator.call_count == 3

        # Check that all three evaluator types were created
        evaluators = [call[0][0] for call in dataset.add_evaluator.call_args_list]
        types = [type(ev) for ev in evaluators]

        assert ScalarEquals in types
        assert ListRecall in types
        assert ValueInExpectedList in types

    def test_unknown_evaluator_kind(self) -> None:
        """Test error handling for unknown evaluator kinds."""
        dataset = Mock()
        specs = [
            {"kind": "UnknownEvaluator", "output_path": "test", "expected_path": "test"}
        ]

        with pytest.raises(
            ValueError, match="Unknown evaluator kind: UnknownEvaluator"
        ):
            from_specs(dataset, specs)

    def test_spec_without_name(self) -> None:
        """Test creating evaluator without explicit name."""
        dataset = Mock()
        specs = [
            {"kind": "ScalarEquals", "output_path": "test", "expected_path": "test"}
        ]

        from_specs(dataset, specs)

        evaluator = dataset.add_evaluator.call_args[0][0]
        # Should not have evaluation_name set (or should be None)
        assert evaluator.evaluation_name is None

    def test_spec_without_normalize(self) -> None:
        """Test creating evaluator without normalize options."""
        dataset = Mock()
        specs = [{"kind": "ListRecall", "output_path": "test", "expected_path": "test"}]

        from_specs(dataset, specs)

        evaluator = dataset.add_evaluator.call_args[0][0]
        assert evaluator.normalize_opts is None

    def test_spec_with_extra_parameters(self) -> None:
        """Test that extra parameters are passed through."""
        dataset = Mock()
        specs = [
            {
                "kind": "ListRecall",
                "output_path": "test",
                "expected_path": "test",
                "multiset": True,
                "element_coerce_to": "int",
            }
        ]

        from_specs(dataset, specs)

        evaluator = dataset.add_evaluator.call_args[0][0]
        assert evaluator.multiset is True
        assert evaluator.element_coerce_to == "int"

    def test_spec_modification_safety(self) -> None:
        """Test that original specs are not modified."""
        dataset = Mock()
        original_spec = {
            "kind": "ScalarEquals",
            "output_path": "test",
            "expected_path": "test",
            "name": "test_eval",
            "normalize": {"lowercase": True},
        }

        # Make a copy to track changes
        spec_copy = dict(original_spec)

        from_specs(dataset, [original_spec])

        # Original spec should be unchanged
        assert original_spec == spec_copy
        assert "kind" in original_spec
        assert "name" in original_spec
        assert "normalize" in original_spec

    def test_empty_specs_list(self) -> None:
        """Test handling of empty specs list."""
        dataset = Mock()

        from_specs(dataset, [])

        dataset.add_evaluator.assert_not_called()

    def test_all_supported_evaluator_kinds(self) -> None:
        """Test that all documented evaluator kinds work."""
        dataset = Mock()
        specs = [
            {"kind": "CompareFields", "output_path": "test", "expected_path": "test"},
            {"kind": "ScalarEquals", "output_path": "test", "expected_path": "test"},
            {"kind": "ListEquality", "output_path": "test", "expected_path": "test"},
            {"kind": "ListRecall", "output_path": "test", "expected_path": "test"},
            {"kind": "ListPrecision", "output_path": "test", "expected_path": "test"},
            {
                "kind": "ValueInExpectedList",
                "output_path": "test",
                "expected_path": "test",
            },
        ]

        from_specs(dataset, specs)

        assert dataset.add_evaluator.call_count == 6

        # Check that all evaluator types were created successfully
        evaluators = [call[0][0] for call in dataset.add_evaluator.call_args_list]

        assert len(evaluators) == 6
        # All should be instances of some evaluator class
        for evaluator in evaluators:
            # They should all have the basic attributes
            assert hasattr(evaluator, "output_path")
            assert hasattr(evaluator, "expected_path")
            assert hasattr(evaluator, "evaluate")

    def test_complex_realistic_specs(self) -> None:
        """Test a realistic complex specification."""
        dataset = Mock()
        specs = [
            {
                "kind": "ScalarEquals",
                "name": "price_accuracy",
                "output_path": "predicted.price",
                "expected_path": "actual.price",
                "coerce_to": "float",
                "abs_tol": 0.01,
                "rel_tol": 0.05,
            },
            {
                "kind": "ListRecall",
                "name": "tag_coverage",
                "output_path": "predicted.tags",
                "expected_path": "required.tags",
                "normalize": {"lowercase": True, "strip": True, "alphanum": True},
                "multiset": False,
            },
            {
                "kind": "ValueInExpectedList",
                "name": "category_validation",
                "output_path": "predicted.category",
                "expected_path": "valid.categories",
                "normalize": {"lowercase": True},
                "element_coerce_to": "str",
            },
        ]

        from_specs(dataset, specs)

        assert dataset.add_evaluator.call_count == 3

        evaluators = [call[0][0] for call in dataset.add_evaluator.call_args_list]

        # Check first evaluator (ScalarEquals)
        price_eval = evaluators[0]
        assert isinstance(price_eval, ScalarEquals)
        assert price_eval.evaluation_name == "price_accuracy"
        assert price_eval.output_path == "predicted.price"
        assert price_eval.coerce_to == "float"
        assert price_eval.abs_tol == 0.01

        # Check second evaluator (ListRecall)
        tag_eval = evaluators[1]
        assert isinstance(tag_eval, ListRecall)
        assert tag_eval.evaluation_name == "tag_coverage"
        assert tag_eval.normalize_lowercase
        assert tag_eval.normalize_strip
        assert tag_eval.normalize_alphanum
        assert tag_eval.multiset is False

        # Check third evaluator (ValueInExpectedList)
        category_eval = evaluators[2]
        assert isinstance(category_eval, ValueInExpectedList)
        assert category_eval.evaluation_name == "category_validation"
        assert category_eval.element_coerce_to == "str"
