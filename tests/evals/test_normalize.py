"""Test suite for normalization and fuzzy matching utilities."""

import pytest
from pydantic_ai_helpers.evals.normalize import (
    CompareOptions,
    FuzzyOptions,
    NormalizeOptions,
    fuzzy_match_score,
    fuzzy_match_with_options,
    maybe_text_normalize,
    maybe_text_normalize_with_options,
    normalize_iter,
    text_normalize,
    text_normalize_with_options,
)


class TestTextNormalize:
    """Test the text_normalize function."""

    def test_no_normalization(self) -> None:
        """Test that text is unchanged when no options are specified."""
        text = "  Hello World!  "
        result = text_normalize(text)
        assert result == text

    def test_lowercase(self) -> None:
        """Test lowercase normalization."""
        assert text_normalize("Hello World", lowercase=True) == "hello world"
        assert text_normalize("UPPER", lowercase=True) == "upper"
        assert text_normalize("MiXeD", lowercase=True) == "mixed"
        assert text_normalize("", lowercase=True) == ""

    def test_strip(self) -> None:
        """Test whitespace stripping."""
        assert text_normalize("  hello  ", strip=True) == "hello"
        assert text_normalize("\t\nworld\t\n", strip=True) == "world"
        assert text_normalize("no-spaces", strip=True) == "no-spaces"
        assert text_normalize("", strip=True) == ""
        assert text_normalize("   ", strip=True) == ""

    def test_alphanum(self) -> None:
        """Test alphanumeric-only normalization."""
        assert text_normalize("hello-world", alphanum=True) == "helloworld"
        assert text_normalize("test@123!#", alphanum=True) == "test123"
        assert text_normalize("spaces  here", alphanum=True) == "spaceshere"
        assert text_normalize("123", alphanum=True) == "123"
        assert text_normalize("", alphanum=True) == ""
        assert text_normalize("!@#$%", alphanum=True) == ""

    def test_collapse_spaces(self) -> None:
        """Test space collapsing normalization."""
        assert text_normalize("hello   world", collapse_spaces=True) == "hello world"
        assert (
            text_normalize("multiple    spaces   here", collapse_spaces=True)
            == "multiple spaces here"
        )
        assert (
            text_normalize("\t\nwhitespace\t\n", collapse_spaces=True) == "whitespace"
        )
        assert (
            text_normalize("   leading and trailing   ", collapse_spaces=True)
            == "leading and trailing"
        )
        assert text_normalize("", collapse_spaces=True) == ""

    def test_combined_normalizations(self) -> None:
        """Test combinations of normalization options."""
        # lowercase + strip
        result = text_normalize("  HELLO WORLD  ", lowercase=True, strip=True)
        assert result == "hello world"

        # lowercase + alphanum
        result = text_normalize("Hello-World!", lowercase=True, alphanum=True)
        assert result == "helloworld"

        # strip + collapse_spaces
        result = text_normalize("  hello   world  ", strip=True, collapse_spaces=True)
        assert result == "hello world"

        # All options
        result = text_normalize(
            "  HELLO---WORLD!!!  ",
            lowercase=True,
            strip=True,
            alphanum=True,
            collapse_spaces=True,
        )
        assert result == "helloworld"

    def test_order_independence(self) -> None:
        """Test that normalization order produces consistent results."""
        text = "  HELLO-WORLD!  "

        # Different orders should produce same result
        result1 = text_normalize(text, lowercase=True, strip=True, alphanum=True)
        result2 = text_normalize(text, strip=True, alphanum=True, lowercase=True)
        result3 = text_normalize(text, alphanum=True, lowercase=True, strip=True)

        assert result1 == result2 == result3 == "helloworld"

    def test_unicode_handling(self) -> None:
        """Test normalization with Unicode characters."""
        # Basic Unicode
        assert text_normalize("Héllo Wörld", lowercase=True) == "héllo wörld"

        # Emojis should be removed with alphanum
        assert text_normalize("Hello 🌍 World", alphanum=True) == "HelloWorld"

        # Mixed Unicode and ASCII
        assert text_normalize("  Café ☕  ", strip=True, lowercase=True) == "café ☕"


class TestMaybeTextNormalize:
    """Test the maybe_text_normalize function."""

    def test_string_normalization(self) -> None:
        """Test that strings are normalized."""
        result = maybe_text_normalize("  HELLO  ", lowercase=True, strip=True)
        assert result == "hello"

    def test_non_string_passthrough(self) -> None:
        """Test that non-strings are passed through unchanged."""
        assert maybe_text_normalize(42, lowercase=True) == 42
        assert maybe_text_normalize([1, 2, 3], strip=True) == [1, 2, 3]
        assert maybe_text_normalize(None, alphanum=True) is None
        assert maybe_text_normalize({"key": "value"}, lowercase=True) == {
            "key": "value"
        }

    def test_empty_string(self) -> None:
        """Test empty string handling."""
        assert maybe_text_normalize("", lowercase=True) == ""
        assert maybe_text_normalize("", strip=True) == ""

    def test_with_all_options(self) -> None:
        """Test with all normalization options."""
        text = "  HELLO-WORLD!  "
        result = maybe_text_normalize(
            text, lowercase=True, strip=True, alphanum=True, collapse_spaces=True
        )
        assert result == "helloworld"

    def test_type_preservation(self) -> None:
        """Test that non-string types are preserved exactly."""
        inputs = [0, 1, -1, 3.14, True, False, [], {}, set()]
        for inp in inputs:
            result = maybe_text_normalize(
                inp, lowercase=True, strip=True, alphanum=True
            )
            assert result == inp
            assert type(result) is type(inp)


class TestNormalizeIter:
    """Test the normalize_iter function."""

    def test_no_normalizer(self) -> None:
        """Test that elements are unchanged when no normalizer is provided."""
        items = ["hello", "world", 123, None]
        result = normalize_iter(items)
        assert result == items

    def test_none_normalizer(self) -> None:
        """Test explicit None normalizer."""
        items = ["hello", "world"]
        result = normalize_iter(items, element_normalizer=None)
        assert result == items

    def test_with_normalizer(self) -> None:
        """Test with a custom normalizer function."""
        items = ["  HELLO  ", "  WORLD  ", "  TEST  "]
        result = normalize_iter(items, element_normalizer=lambda x: x.strip().lower())
        assert result == ["hello", "world", "test"]

    def test_mixed_types(self) -> None:
        """Test normalizer with mixed types."""
        items = ["  HELLO  ", 42, "  WORLD  ", None]

        def safe_normalizer(x):
            if isinstance(x, str):
                return x.strip().lower()
            return x

        result = normalize_iter(items, element_normalizer=safe_normalizer)
        assert result == ["hello", 42, "world", None]

    def test_empty_iterable(self) -> None:
        """Test with empty iterable."""
        result = normalize_iter([], element_normalizer=lambda x: x.upper())
        assert result == []

    def test_generator_input(self) -> None:
        """Test with generator input."""

        def gen():
            yield "hello"
            yield "world"

        result = normalize_iter(gen(), element_normalizer=lambda x: x.upper())
        assert result == ["HELLO", "WORLD"]

    def test_tuple_input(self) -> None:
        """Test with tuple input (returns list)."""
        items = ("hello", "world")
        result = normalize_iter(items, element_normalizer=lambda x: x.upper())
        assert result == ["HELLO", "WORLD"]
        assert isinstance(result, list)

    def test_normalizer_exceptions(self) -> None:
        """Test behavior when normalizer raises exceptions."""
        items = ["hello", "world"]

        def failing_normalizer(x):
            if x == "world":
                raise ValueError("Test error")
            return x.upper()

        # The function should not catch exceptions - they should propagate
        with pytest.raises(ValueError, match="Test error"):
            normalize_iter(items, element_normalizer=failing_normalizer)

    def test_complex_normalizer(self) -> None:
        """Test with a complex normalizer that uses text_normalize."""
        items = ["  Hello-World!  ", "  FOO@BAR  ", "  test123  "]

        def complex_normalizer(x):
            return text_normalize(x, lowercase=True, strip=True, alphanum=True)

        result = normalize_iter(items, element_normalizer=complex_normalizer)
        assert result == ["helloworld", "foobar", "test123"]


class TestNormalizeOptions:
    """Test the NormalizeOptions dataclass."""

    def test_defaults(self) -> None:
        """Test default values for NormalizeOptions."""
        opts = NormalizeOptions()
        assert opts.lowercase is True
        assert opts.strip is True
        assert opts.collapse_spaces is True
        assert opts.alphanum is False

    def test_custom_values(self) -> None:
        """Test custom values for NormalizeOptions."""
        opts = NormalizeOptions(
            lowercase=False,
            strip=False,
            collapse_spaces=False,
            alphanum=True,
        )
        assert opts.lowercase is False
        assert opts.strip is False
        assert opts.collapse_spaces is False
        assert opts.alphanum is True


class TestTextNormalizeWithOptions:
    """Test the text_normalize_with_options function."""

    def test_default_options(self) -> None:
        """Test normalization with default options."""
        opts = NormalizeOptions()
        text = "  HELLO World!  "
        result = text_normalize_with_options(text, opts)
        # Default: lowercase=True, strip=True, collapse_spaces=True, alphanum=False
        assert result == "hello world!"

    def test_custom_options(self) -> None:
        """Test normalization with custom options."""
        opts = NormalizeOptions(lowercase=False, strip=False, alphanum=True)
        text = "  HELLO-World_123!  "
        result = text_normalize_with_options(text, opts)
        # Only alphanum=True applied
        assert result == "  HELLOWorld123  "

    def test_maybe_normalize_with_options(self) -> None:
        """Test maybe_text_normalize_with_options function."""
        opts = NormalizeOptions()

        # String should be normalized
        result = maybe_text_normalize_with_options("  HELLO  ", opts)
        assert result == "hello"

        # Non-string should be unchanged
        result = maybe_text_normalize_with_options(123, opts)
        assert result == 123


class TestFuzzyOptions:
    """Test the FuzzyOptions dataclass."""

    def test_defaults(self) -> None:
        """Test default values for FuzzyOptions."""
        opts = FuzzyOptions()
        assert opts.enabled is True
        assert opts.threshold == 0.85
        assert opts.algorithm == "token_set_ratio"

    def test_custom_values(self) -> None:
        """Test custom values for FuzzyOptions."""
        opts = FuzzyOptions(
            enabled=False,
            threshold=0.9,
            algorithm="ratio",
        )
        assert opts.enabled is False
        assert opts.threshold == 0.9
        assert opts.algorithm == "ratio"


class TestCompareOptions:
    """Test the CompareOptions dataclass."""

    def test_defaults(self) -> None:
        """Test default values for CompareOptions."""
        opts = CompareOptions()
        assert opts.normalize is None
        assert opts.fuzzy is None
        assert opts.coerce_to is None
        assert opts.abs_tol is None
        assert opts.rel_tol is None
        assert opts.enum_values is None

    def test_with_sub_options(self) -> None:
        """Test CompareOptions with sub-options."""
        normalize_opts = NormalizeOptions(lowercase=False)
        fuzzy_opts = FuzzyOptions(threshold=0.9)

        opts = CompareOptions(
            normalize=normalize_opts,
            fuzzy=fuzzy_opts,
            coerce_to="str",
            abs_tol=0.1,
        )

        assert opts.normalize is normalize_opts
        assert opts.fuzzy is fuzzy_opts
        assert opts.coerce_to == "str"
        assert opts.abs_tol == 0.1


class TestFuzzyMatching:
    """Test fuzzy matching functionality."""

    def test_fuzzy_match_score_exact(self) -> None:
        """Test fuzzy matching with exact matches."""
        score = fuzzy_match_score("hello", "hello")
        assert score == 1.0

        score = fuzzy_match_score("hello world", "hello world")
        assert score == 1.0

    def test_fuzzy_match_score_different_algorithms(self) -> None:
        """Test different fuzzy matching algorithms."""
        s1, s2 = "hello world", "hello wrld"

        # All algorithms should return scores between 0 and 1
        algorithms = ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"]
        for algorithm in algorithms:
            score = fuzzy_match_score(s1, s2, algorithm)
            assert 0.0 <= score <= 1.0

    def test_fuzzy_match_score_no_match(self) -> None:
        """Test fuzzy matching with completely different strings."""
        score = fuzzy_match_score("hello", "xyz123")
        assert score < 0.5  # Should be a low score

    def test_fuzzy_match_with_options_defaults(self) -> None:
        """Test fuzzy_match_with_options with default options."""
        # Exact match after normalization
        score, is_match = fuzzy_match_with_options("  HELLO  ", "hello")
        assert score == 1.0
        assert is_match is True

        # Close match
        score, is_match = fuzzy_match_with_options("hello", "helo")
        assert 0.7 < score < 1.0
        # Should be a match if score >= 0.85 (default threshold)

    def test_fuzzy_match_with_options_custom_threshold(self) -> None:
        """Test fuzzy_match_with_options with custom threshold."""
        fuzzy_opts = FuzzyOptions(threshold=0.95)

        score, is_match = fuzzy_match_with_options(
            "hello", "helo", fuzzy_options=fuzzy_opts
        )
        # Score should be decent but below 0.95
        assert 0.7 < score < 0.95
        assert is_match is False

    def test_fuzzy_match_with_options_disabled(self) -> None:
        """Test fuzzy_match_with_options with fuzzy disabled."""
        fuzzy_opts = FuzzyOptions(enabled=False)

        # Exact match after normalization should work
        score, is_match = fuzzy_match_with_options(
            "  HELLO  ", "hello", fuzzy_options=fuzzy_opts
        )
        assert score == 1.0
        assert is_match is True

        # Non-exact should fail
        score, is_match = fuzzy_match_with_options(
            "hello", "helo", fuzzy_options=fuzzy_opts
        )
        assert score == 0.0
        assert is_match is False

    def test_fuzzy_match_normalization_before_fuzzy(self) -> None:
        """Test that normalization happens before fuzzy matching."""
        # These should match perfectly after normalization
        score, is_match = fuzzy_match_with_options("  HELLO WORLD  ", "hello    world")
        assert score == 1.0
        assert is_match is True
