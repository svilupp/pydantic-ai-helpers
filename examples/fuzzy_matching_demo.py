#!/usr/bin/env python3
"""
Fuzzy Matching Demo for pydantic-ai-helpers evals

This script demonstrates the new fuzzy string matching capabilities
for AI evaluation tasks.
"""

from pydantic_ai_helpers.evals import (
    CompareOptions,
    FuzzyOptions,
    ListCompare,
    ListRecall,
    NormalizeOptions,
    ScalarCompare,
    ScalarEquals,
    ValueInExpectedList,
)
from pydantic_evals.evaluators import EvaluatorContext


def basic_fuzzy_examples():
    """Basic fuzzy matching examples."""
    print("=== Basic Fuzzy Matching ===\n")

    # Default fuzzy matching
    comp = ScalarCompare()

    test_cases = [
        ("color", "colour"),
        ("New York", "new york"),
        ("iPhone", "iphone"),
        ("machine learning", "Machine Learning"),
        ("apple", "aple"),  # typo
        ("completely", "different"),  # no match
    ]

    for s1, s2 in test_cases:
        score, reason = comp(s1, s2)
        match_status = "✓ MATCH" if score >= 0.85 else "✗ NO MATCH"
        print(f"{s1:15} vs {s2:15} → {score:.3f} {match_status}")
        print(f"  Reason: {reason}\n")


def algorithm_comparison():
    """Compare different fuzzy algorithms."""
    print("=== Fuzzy Algorithm Comparison ===\n")

    algorithms = ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"]
    test_cases = [
        ("New York City", "NYC"),
        ("machine learning", "ML algorithms"),
        ("iPhone 15 Pro", "Apple iPhone 15 Pro Max"),
        ("data science", "Data Science & Analytics"),
    ]

    for s1, s2 in test_cases:
        print(f"Comparing: '{s1}' vs '{s2}'")
        for algorithm in algorithms:
            comp = ScalarCompare(
                options=CompareOptions(
                    normalize=NormalizeOptions(lowercase=True),
                    fuzzy=FuzzyOptions(enabled=True, algorithm=algorithm),
                )
            )
            score, _ = comp(s1, s2)
            print(f"  {algorithm:20}: {score:.3f}")
        print()


def list_fuzzy_examples():
    """Fuzzy matching with lists."""
    print("=== List Fuzzy Matching ===\n")

    # List recall with fuzzy matching
    recall_comp = ListCompare(mode="recall")

    # Simulated AI output vs expected tags
    ai_tags = ["Python", "AI", "Machine Learning", "Data Science"]
    expected_tags = ["python", "artificial intelligence", "ML", "data analysis"]

    score, reason = recall_comp(ai_tags, expected_tags)
    print(f"AI Tags: {ai_tags}")
    print(f"Expected: {expected_tags}")
    print(f"Fuzzy Recall Score: {score:.3f}")
    print(f"Reason: {reason}\n")

    # Compare with exact matching
    exact_comp = ListCompare(
        mode="recall", options=CompareOptions(fuzzy=FuzzyOptions(enabled=False))
    )
    exact_score, exact_reason = exact_comp(ai_tags, expected_tags)
    print(f"Exact Recall Score: {exact_score:.3f}")
    print(f"Reason: {exact_reason}\n")


def real_world_ai_evaluation():
    """Real-world AI evaluation scenarios."""
    print("=== Real-world AI Evaluation ===\n")

    # Product name evaluation
    product_eval = ScalarEquals(
        output_path="product_name",
        expected_path="product_name",
        fuzzy_threshold=0.8,
        normalize_lowercase=True,
        evaluation_name="product_match",
    )

    # AI generated product name vs expected
    ctx = EvaluatorContext(
        name="test_context",
        inputs=None,
        metadata=None,
        output={"product_name": "iPhone 15 Pro Max 256GB Titanium"},
        expected_output={"product_name": "iPhone 15 Pro Max 256 GB titanium"},
        duration=0.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )

    result = product_eval.evaluate(ctx)
    print("Product Name Evaluation:")
    print("  AI Output: 'iPhone 15 Pro Max 256GB Titanium'")
    print("  Expected:  'iPhone 15 Pro Max 256 GB titanium'")
    print(f"  Score: {result.value:.3f}")
    print(f"  Reason: {result.reason}\n")

    # Tag classification evaluation
    tag_eval = ListRecall(
        output_path="ai_tags",
        expected_path="human_tags",
        normalize_strip=True,
        fuzzy_threshold=0.7,
        evaluation_name="tag_recall",
    )

    ctx = EvaluatorContext(
        name="test_context",
        inputs=None,
        metadata=None,
        output={
            "ai_tags": [
                "Machine Learning",
                "Artificial Intelligence",
                "Python Programming",
            ]
        },
        expected_output={
            "human_tags": ["ML", "AI", "programming", "data science", "algorithms"]
        },
        duration=0.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )

    result = tag_eval.evaluate(ctx)
    print("Tag Classification Evaluation:")
    print(f"  AI Tags:   {ctx.output['ai_tags']}")
    print(f"  Expected:  {ctx.expected_output['human_tags']}")
    print(f"  Score: {result.value:.3f}")
    print(f"  Reason: {result.reason}\n")


def category_validation_example():
    """Category validation with fuzzy fallback."""
    print("=== Category Validation ===\n")

    category_eval = ValueInExpectedList(
        output_path="predicted_category",
        expected_path="valid_categories",
        normalize_alphanum=True,
        fuzzy_threshold=0.8,
        evaluation_name="category_check",
    )

    test_cases = [
        {
            "predicted": "Technology & Programming",
            "valid": ["Technology", "Science", "Business", "Education"],
            "description": "Close match with punctuation",
        },
        {
            "predicted": "Artificial Intelligence",
            "valid": ["AI", "Machine Learning", "Data Science", "Programming"],
            "description": "AI abbreviation vs full name",
        },
        {
            "predicted": "Web Development",
            "valid": ["Frontend", "Backend", "Database", "API"],
            "description": "No good match",
        },
    ]

    for case in test_cases:
        ctx = EvaluatorContext(
            name="test_context",
            inputs=None,
            metadata=None,
            output={"predicted_category": case["predicted"]},
            expected_output={"valid_categories": case["valid"]},
            duration=0.0,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        result = category_eval.evaluate(ctx)
        print(f"{case['description']}:")
        print(f"  Predicted: '{case['predicted']}'")
        print(f"  Valid:     {case['valid']}")
        print(f"  Score:     {result.value:.3f}")
        print(f"  Reason:    {result.reason}\n")


def threshold_sensitivity_demo():
    """Demonstrate threshold sensitivity."""
    print("=== Threshold Sensitivity ===\n")

    test_pair = ("machine learning", "ML algorithms")
    thresholds = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]

    print(f"Comparing: '{test_pair[0]}' vs '{test_pair[1]}'")
    print("Threshold | Score | Match?")
    print("-" * 30)

    for threshold in thresholds:
        comp = ScalarCompare(
            options=CompareOptions(
                normalize=NormalizeOptions(lowercase=True),
                fuzzy=FuzzyOptions(enabled=True, threshold=threshold),
            )
        )
        score, _ = comp(test_pair[0], test_pair[1])
        is_match = "✓" if score >= threshold else "✗"
        print(f"   {threshold:4.2f}   | {score:.3f} |   {is_match}")


def advanced_options_demo():
    """Advanced fuzzy options demonstration."""
    print("\n=== Advanced Options ===\n")

    # Custom options with strict normalization
    opts = CompareOptions(
        normalize=NormalizeOptions(
            lowercase=True,
            strip=True,
            alphanum=True,  # Remove all punctuation
            collapse_spaces=True,
        ),
        fuzzy=FuzzyOptions(enabled=True, threshold=0.8, algorithm="token_set_ratio"),
    )

    evaluator = ScalarEquals(
        output_path="description",
        expected_path="description",
        compare_options=opts,
        evaluation_name="advanced_match",
    )

    ctx = EvaluatorContext(
        name="test_context",
        inputs=None,
        metadata=None,
        output={"description": "High-Performance, AI-Powered Analytics Platform!!!"},
        expected_output={
            "description": "high performance ai powered analytics platform"
        },
        duration=0.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )

    result = evaluator.evaluate(ctx)
    print("Advanced normalization example:")
    print("  Input:    'High-Performance, AI-Powered Analytics Platform!!!'")
    print("  Expected: 'high performance ai powered analytics platform'")
    print(f"  Score:    {result.value:.3f}")
    print(f"  Reason:   {result.reason}")


if __name__ == "__main__":
    print("🔍 Fuzzy Matching Demo for pydantic-ai-helpers\n")

    basic_fuzzy_examples()
    algorithm_comparison()
    list_fuzzy_examples()
    real_world_ai_evaluation()
    category_validation_example()
    threshold_sensitivity_demo()
    advanced_options_demo()

    print(
        "\n✨ Demo complete! Try experimenting with different thresholds and algorithms."
    )
