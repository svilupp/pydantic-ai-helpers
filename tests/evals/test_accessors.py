"""Test suite for accessor utilities."""

from dataclasses import dataclass

import pytest
from pydantic_ai_helpers.evals.accessors import Accessor, resolve_path


@dataclass
class SampleData:
    """Sample dataclass for testing attribute access."""

    name: str
    age: int
    scores: list[int]


class TestResolvePath:
    """Test the resolve_path function with various input types."""

    def test_empty_path(self) -> None:
        """Test that empty or None paths return the root object."""
        obj = {"test": "value"}

        ok, value, reason = resolve_path(obj, None)
        assert ok is True
        assert value == obj
        assert reason is None

        ok, value, reason = resolve_path(obj, "")
        assert ok is True
        assert value == obj
        assert reason is None

        ok, value, reason = resolve_path(obj, ".")
        assert ok is True
        assert value == obj
        assert reason is None

    def test_dict_access(self) -> None:
        """Test accessing dictionary keys."""
        obj = {
            "user": {"name": "Alice", "profile": {"age": 30, "scores": [10, 20, 30]}}
        }

        # Simple key access
        ok, value, reason = resolve_path(obj, "user")
        assert ok is True
        assert value == obj["user"]
        assert reason is None

        # Nested key access
        ok, value, reason = resolve_path(obj, "user.name")
        assert ok is True
        assert value == "Alice"
        assert reason is None

        # Deep nested access
        ok, value, reason = resolve_path(obj, "user.profile.age")
        assert ok is True
        assert value == 30
        assert reason is None

    def test_attribute_access(self) -> None:
        """Test accessing object attributes."""
        obj = SampleData(name="Bob", age=25, scores=[1, 2, 3])

        ok, value, reason = resolve_path(obj, "name")
        assert ok is True
        assert value == "Bob"
        assert reason is None

        ok, value, reason = resolve_path(obj, "age")
        assert ok is True
        assert value == 25
        assert reason is None

        ok, value, reason = resolve_path(obj, "scores")
        assert ok is True
        assert value == [1, 2, 3]
        assert reason is None

    def test_sequence_access(self) -> None:
        """Test accessing sequence indices."""
        obj = {
            "users": [
                {"name": "Alice", "scores": [10, 20]},
                {"name": "Bob", "scores": [30, 40]},
            ]
        }

        # Access list element
        ok, value, reason = resolve_path(obj, "users.0")
        assert ok is True
        assert value == {"name": "Alice", "scores": [10, 20]}
        assert reason is None

        # Access nested sequence
        ok, value, reason = resolve_path(obj, "users.1.scores.0")
        assert ok is True
        assert value == 30
        assert reason is None

        # Negative indices
        ok, value, reason = resolve_path(obj, "users.-1.name")
        assert ok is True
        assert value == "Bob"
        assert reason is None

    def test_mixed_access(self) -> None:
        """Test mixing dict, attribute, and sequence access."""
        data = SampleData(name="Test", age=42, scores=[100, 200])
        obj = {"data": data, "items": [data, {"nested": data}]}

        # Dict -> attribute
        ok, value, reason = resolve_path(obj, "data.name")
        assert ok is True
        assert value == "Test"
        assert reason is None

        # Dict -> sequence -> attribute
        ok, value, reason = resolve_path(obj, "items.0.age")
        assert ok is True
        assert value == 42
        assert reason is None

        # Dict -> sequence -> dict -> attribute -> sequence
        ok, value, reason = resolve_path(obj, "items.1.nested.scores.1")
        assert ok is True
        assert value == 200
        assert reason is None

    def test_missing_key_error(self) -> None:
        """Test error handling for missing keys."""
        obj = {"user": {"name": "Alice"}}

        ok, value, reason = resolve_path(obj, "missing")
        assert ok is False
        assert value is None
        assert "missing key/attr 'missing'" in reason

        ok, value, reason = resolve_path(obj, "user.missing")
        assert ok is False
        assert value is None
        assert "missing key/attr 'missing'" in reason

    def test_missing_attribute_error(self) -> None:
        """Test error handling for missing attributes."""
        obj = SampleData(name="Test", age=30, scores=[])

        ok, value, reason = resolve_path(obj, "missing_attr")
        assert ok is False
        assert value is None
        assert "missing key/attr 'missing_attr'" in reason

    def test_index_error(self) -> None:
        """Test error handling for invalid indices."""
        obj = {"items": [1, 2, 3]}

        # Index out of range
        ok, value, reason = resolve_path(obj, "items.10")
        assert ok is False
        assert value is None
        assert "error at '10'" in reason

        # Index on non-indexable
        ok, value, reason = resolve_path(obj, "items.0.5")
        assert ok is False
        assert value is None
        assert "requires indexable type" in reason

    def test_none_traversal_error(self) -> None:
        """Test error handling when traversing through None."""
        obj = {"user": None}

        ok, value, reason = resolve_path(obj, "user.name")
        assert ok is False
        assert value is None
        assert "segment 'name' on None" in reason

    def test_non_indexable_error(self) -> None:
        """Test error handling for non-indexable objects."""
        obj = {"value": 42}  # Integer is not indexable

        ok, value, reason = resolve_path(obj, "value.0")
        assert ok is False
        assert value is None
        assert "requires indexable type" in reason


class TestAccessor:
    """Test the Accessor class."""

    def test_accessor_with_path(self) -> None:
        """Test Accessor with a configured path."""
        accessor = Accessor("user.name")
        obj = {"user": {"name": "Alice", "age": 30}}

        ok, value, reason = accessor.get(obj)
        assert ok is True
        assert value == "Alice"
        assert reason is None

    def test_accessor_without_path(self) -> None:
        """Test Accessor with no path (returns root)."""
        accessor = Accessor()
        obj = {"test": "value"}

        ok, value, reason = accessor.get(obj)
        assert ok is True
        assert value == obj
        assert reason is None

    def test_accessor_with_none_path(self) -> None:
        """Test Accessor with explicit None path."""
        accessor = Accessor(path=None)
        obj = {"test": "value"}

        ok, value, reason = accessor.get(obj)
        assert ok is True
        assert value == obj
        assert reason is None

    def test_accessor_error_propagation(self) -> None:
        """Test that Accessor properly propagates errors."""
        accessor = Accessor("missing.key")
        obj = {"user": {"name": "Alice"}}

        ok, value, reason = accessor.get(obj)
        assert ok is False
        assert value is None
        assert "missing key/attr 'missing'" in reason

    def test_accessor_immutability(self) -> None:
        """Test that Accessor is properly frozen/immutable."""
        accessor = Accessor("test.path")

        # Should not be able to modify the path
        with pytest.raises(
            (AttributeError, TypeError)
        ):  # Either AttributeError or dataclass FrozenInstanceError
            accessor.path = "new.path"  # type: ignore

    def test_accessor_repr(self) -> None:
        """Test Accessor string representation."""
        accessor = Accessor("user.name")
        repr_str = repr(accessor)

        # Should contain the class name and path
        assert "Accessor" in repr_str
        assert "user.name" in repr_str
