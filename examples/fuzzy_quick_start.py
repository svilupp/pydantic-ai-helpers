#!/usr/bin/env python3
"""
Quick Start Guide: Fuzzy Matching with pydantic-ai-helpers

This script shows the most common fuzzy matching patterns you'll need
for AI evaluation tasks.
"""

from pydantic_ai_helpers.evals import (
    ListCompare,
    ListRecall,
    ScalarCompare,
    ScalarEquals,
    ValueInExpectedList,
)
from pydantic_evals.evaluators import EvaluatorContext


def main():
    print("🚀 Fuzzy Matching Quick Start\n")

    # 1. Basic String Comparison with Fuzzy Matching
    print("1️⃣ Basic String Comparison")
    print("-" * 30)

    # Default: fuzzy enabled with 0.85 threshold
    comp = ScalarCompare()

    examples = [
        ("color", "colour"),  # Spelling variation
        ("New York", "new york"),  # Case difference
        ("AI", "artificial intelligence"),  # Abbreviation
        ("hello", "helo"),  # Typo
    ]

    for s1, s2 in examples:
        score, reason = comp(s1, s2)
        print(f"'{s1}' vs '{s2}' → {score:.3f}")

    print()

    # 2. List Evaluation with Fuzzy Matching
    print("2️⃣ List Evaluation")
    print("-" * 20)

    # How many expected tags did the AI capture?
    recall_comp = ListCompare(mode="recall")

    ai_tags = ["Python", "AI", "Machine Learning"]
    expected_tags = ["python", "artificial intelligence", "ML", "data science"]

    score, reason = recall_comp(ai_tags, expected_tags)
    print(f"AI Tags: {ai_tags}")
    print(f"Expected: {expected_tags}")
    print(f"Recall Score: {score:.3f}")
    print(f"Reason: {reason}")
    print()

    # 3. Field-to-Field Evaluation
    print("3️⃣ Field-to-Field Evaluation")
    print("-" * 30)

    # Evaluate nested object fields
    evaluator = ScalarEquals(
        output_path="product.name",
        expected_path="product.name",
        fuzzy_threshold=0.8,  # Allow some flexibility
        evaluation_name="product_name",
    )

    ctx = EvaluatorContext(
        name="test_context",
        inputs=None,
        metadata=None,
        output={"product": {"name": "iPhone 15 Pro Max 256GB"}},
        expected_output={
            "product": {"name": "iPhone 15 Pro Max 256 GB"}
        },  # Space difference
        duration=0.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )

    result = evaluator.evaluate(ctx)
    print("Product name comparison:")
    print(f"Score: {result.value:.3f}")
    print(f"Reason: {result.reason}")
    print()

    # 4. Category Validation
    print("4️⃣ Category Validation")
    print("-" * 22)

    # Check if AI output is in list of valid categories
    validator = ValueInExpectedList(
        output_path="category",
        expected_path="valid_categories",
        fuzzy_threshold=0.85,
        evaluation_name="category_check",
    )

    ctx = EvaluatorContext(
        name="test_context",
        inputs=None,
        metadata=None,
        output={"category": "Machine Learning"},
        expected_output={
            "valid_categories": ["AI", "ML", "Data Science", "Programming"]
        },
        duration=0.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )

    result = validator.evaluate(ctx)
    print("Category: 'Machine Learning'")
    print("Valid options: ['AI', 'ML', 'Data Science', 'Programming']")
    print(f"Validation score: {result.value:.3f}")
    print(f"Reason: {result.reason}")
    print()

    # 5. Practical AI Evaluation Scenario
    print("5️⃣ Real AI Evaluation")
    print("-" * 24)

    # Simulated AI-generated content evaluation
    content_eval = ScalarEquals(
        output_path="title",
        expected_path="title",
        fuzzy_threshold=0.8,
        normalize_lowercase=True,
        evaluation_name="title_match",
    )

    tag_eval = ListRecall(
        output_path="tags",
        expected_path="tags",
        fuzzy_threshold=0.7,
        evaluation_name="tag_recall",
    )

    # AI generated article metadata
    ai_output = {
        "title": "Getting Started with Python for Data Science",
        "tags": ["Python", "Data Science", "Programming", "Tutorial"],
    }

    # Human-annotated ground truth
    expected = {
        "title": "Getting started with Python for data science",  # Case difference
        "tags": ["python", "data analysis", "programming", "education"],  # Variation
    }

    ctx = EvaluatorContext(
        name="test_context",
        inputs=None,
        metadata=None,
        output=ai_output,
        expected_output=expected,
        duration=0.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )

    title_result = content_eval.evaluate(ctx)
    tag_result = tag_eval.evaluate(ctx)

    print("AI-Generated Article Evaluation:")
    print(f"Title match: {title_result.value:.3f} - {title_result.reason}")
    print(f"Tag recall: {tag_result.value:.3f} - {tag_result.reason}")

    # Overall assessment
    avg_score = (title_result.value + tag_result.value) / 2
    print(f"\nOverall Score: {avg_score:.3f}")
    print("✅ Good performance!" if avg_score >= 0.8 else "⚠️  Needs improvement")

    print("\n" + "=" * 50)
    print("🎯 Key Benefits of Fuzzy Matching:")
    print("• Handles typos and spelling variations")
    print("• Case-insensitive comparisons")
    print("• Abbreviation matching (AI ↔ Artificial Intelligence)")
    print("• Flexible thresholds for different use cases")
    print("• Better evaluation of AI-generated content")


if __name__ == "__main__":
    main()
