#!/usr/bin/env python3
"""
AI Evaluation Pipeline Example

This script demonstrates how to build a comprehensive AI evaluation pipeline
using the fuzzy matching capabilities of pydantic-ai-helpers.

It simulates evaluating AI-generated content against human-annotated ground truth,
which is a common pattern in AI system evaluation.
"""

from dataclasses import dataclass
from typing import Any

from pydantic_ai_helpers.evals import (
    ListPrecision,
    ListRecall,
    ScalarEquals,
    ValueInExpectedList,
)
from pydantic_evals.evaluators import EvaluatorContext


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    task_name: str
    evaluator_name: str
    score: float
    reason: str
    threshold_met: bool


class AIEvaluationPipeline:
    """Pipeline for evaluating AI outputs with fuzzy matching."""

    def __init__(self):
        self.evaluators = {}
        self.results = []

    def add_evaluator(self, name: str, evaluator: Any) -> None:
        """Add an evaluator to the pipeline."""
        self.evaluators[name] = evaluator

    def evaluate_sample(
        self, task_name: str, ai_output: dict[str, Any], expected: dict[str, Any]
    ) -> list[EvaluationResult]:
        """Evaluate a single AI output sample."""
        sample_results = []

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

        for evaluator_name, evaluator in self.evaluators.items():
            result = evaluator.evaluate(ctx)

            # Determine if threshold was met (assuming 0.8 is good enough)
            threshold_met = result.value >= 0.8

            eval_result = EvaluationResult(
                task_name=task_name,
                evaluator_name=evaluator_name,
                score=result.value,
                reason=result.reason,
                threshold_met=threshold_met,
            )

            sample_results.append(eval_result)
            self.results.append(eval_result)

        return sample_results

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for all evaluations."""
        if not self.results:
            return {}

        total_evaluations = len(self.results)
        passed_evaluations = sum(1 for r in self.results if r.threshold_met)

        # Group by evaluator
        by_evaluator = {}
        for result in self.results:
            if result.evaluator_name not in by_evaluator:
                by_evaluator[result.evaluator_name] = []
            by_evaluator[result.evaluator_name].append(result.score)

        evaluator_stats = {}
        for name, scores in by_evaluator.items():
            evaluator_stats[name] = {
                "avg_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "pass_rate": sum(1 for s in scores if s >= 0.8) / len(scores),
            }

        return {
            "total_evaluations": total_evaluations,
            "overall_pass_rate": passed_evaluations / total_evaluations,
            "by_evaluator": evaluator_stats,
        }


def setup_content_evaluation_pipeline():
    """Set up a pipeline for evaluating AI-generated content."""
    pipeline = AIEvaluationPipeline()

    # Title evaluation with fuzzy matching for typos
    pipeline.add_evaluator(
        "title_accuracy",
        ScalarEquals(
            output_path="title",
            expected_path="title",
            fuzzy_threshold=0.85,
            normalize_lowercase=True,
            normalize_strip=True,
            evaluation_name="title_match",
        ),
    )

    # Summary evaluation with more lenient fuzzy matching
    pipeline.add_evaluator(
        "summary_similarity",
        ScalarEquals(
            output_path="summary",
            expected_path="summary",
            fuzzy_threshold=0.7,  # More lenient for longer text
            fuzzy_algorithm="token_set_ratio",
            normalize_lowercase=True,
            evaluation_name="summary_match",
        ),
    )

    # Tag recall - how many expected tags were captured
    pipeline.add_evaluator(
        "tag_recall",
        ListRecall(
            output_path="tags",
            expected_path="tags",
            fuzzy_threshold=0.8,
            normalize_lowercase=True,
            evaluation_name="tag_coverage",
        ),
    )

    # Tag precision - how many predicted tags are valid
    pipeline.add_evaluator(
        "tag_precision",
        ListPrecision(
            output_path="tags",
            expected_path="tags",
            fuzzy_threshold=0.8,
            normalize_lowercase=True,
            evaluation_name="tag_accuracy",
        ),
    )

    # Category validation
    pipeline.add_evaluator(
        "category_validation",
        ValueInExpectedList(
            output_path="category",
            expected_path="valid_categories",
            fuzzy_threshold=0.85,
            normalize_alphanum=True,
            evaluation_name="category_check",
        ),
    )

    return pipeline


def run_content_evaluation_demo():
    """Run a demo of content evaluation."""
    print("🤖 AI Content Evaluation Pipeline Demo\n")

    # Set up the pipeline
    pipeline = setup_content_evaluation_pipeline()

    # Simulated AI outputs and ground truth
    test_samples = [
        {
            "task": "news_article_1",
            "ai_output": {
                "title": "Breaking: New AI Model Achieves Human-Level Performance",
                "summary": "Researchers have developed an advanced AI system that matches human capabilities in language understanding and generation tasks.",
                "tags": ["AI", "Machine Learning", "Research", "Technology"],
                "category": "Artificial Intelligence",
            },
            "expected": {
                "title": "Breaking: New AI Model Achieves Human-level Performance",  # Minor typo
                "summary": "Researchers developed an advanced AI system matching human capabilities in language understanding and generation.",
                "tags": [
                    "artificial intelligence",
                    "machine learning",
                    "research",
                    "tech",
                ],  # Slight variations
                "valid_categories": ["AI", "Technology", "Science", "Research"],
            },
        },
        {
            "task": "product_review_1",
            "ai_output": {
                "title": "iPhone 15 Pro Review: Exceptional Camera Quality",
                "summary": "The iPhone 15 Pro delivers outstanding photo and video capabilities with its new camera system.",
                "tags": ["iPhone", "Apple", "smartphone", "camera", "review"],
                "category": "Product Review",
            },
            "expected": {
                "title": "iPhone 15 Pro Review: Exceptional Camera Quality",
                "summary": "iPhone 15 Pro provides excellent photo and video quality thanks to its upgraded camera system.",
                "tags": ["iphone", "apple", "phone", "photography", "product review"],
                "valid_categories": [
                    "Product Review",
                    "Technology",
                    "Mobile",
                    "Consumer Electronics",
                ],
            },
        },
        {
            "task": "tutorial_content_1",
            "ai_output": {
                "title": "Getting Started with Python for Data Science",
                "summary": "Learn the fundamentals of Python programming for data analysis and machine learning applications.",
                "tags": [
                    "Python",
                    "Data Science",
                    "Programming",
                    "Tutorial",
                    "Machine Learning",
                ],
                "category": "Educational Content",
            },
            "expected": {
                "title": "Getting Started with Python for Data Science",
                "summary": "Learn Python programming fundamentals for data analysis and ML applications.",
                "tags": ["python", "data analysis", "programming", "education", "ML"],
                "valid_categories": [
                    "Education",
                    "Programming",
                    "Data Science",
                    "Tutorial",
                ],
            },
        },
    ]

    # Evaluate each sample
    print("Evaluating AI-generated content...\n")

    for sample in test_samples:
        print(f"📝 Task: {sample['task']}")
        print(f"   Title: '{sample['ai_output']['title']}'")
        print(f"   Tags: {sample['ai_output']['tags']}")
        print(f"   Category: '{sample['ai_output']['category']}'")

        results = pipeline.evaluate_sample(
            sample["task"], sample["ai_output"], sample["expected"]
        )

        print("\n   Evaluation Results:")
        for result in results:
            status = "✅ PASS" if result.threshold_met else "❌ FAIL"
            print(f"     {result.evaluator_name:20}: {result.score:.3f} {status}")
            if result.score < 0.8:  # Show reason for failures
                print(f"       Reason: {result.reason}")
        print()

    # Show summary statistics
    stats = pipeline.get_summary_stats()
    print("📊 Summary Statistics:")
    print(f"   Total Evaluations: {stats['total_evaluations']}")
    print(f"   Overall Pass Rate: {stats['overall_pass_rate']:.1%}")
    print("\n   By Evaluator:")

    for evaluator, eval_stats in stats["by_evaluator"].items():
        print(f"     {evaluator:20}:")
        print(f"       Avg Score: {eval_stats['avg_score']:.3f}")
        print(f"       Pass Rate: {eval_stats['pass_rate']:.1%}")
        print(
            f"       Range:     {eval_stats['min_score']:.3f} - {eval_stats['max_score']:.3f}"
        )


def compare_fuzzy_vs_exact():
    """Compare fuzzy vs exact matching performance."""
    print("\n🔍 Fuzzy vs Exact Matching Comparison\n")

    # Test cases with common AI output variations
    test_cases = [
        {
            "ai_output": "Machine Learning and AI Applications",
            "expected": "Machine Learning and Artificial Intelligence Applications",
        },
        {
            "ai_output": "iPhone 15 Pro Max 256GB",
            "expected": "iPhone 15 Pro Max 256 GB",  # Space difference
        },
        {
            "ai_output": "Getting Started with Python Programming",
            "expected": "Getting started with Python programming",  # Case difference
        },
        {
            "ai_output": "Data Science & Analytics",
            "expected": "Data Science and Analytics",  # Punctuation difference
        },
    ]

    # Fuzzy evaluator
    fuzzy_eval = ScalarEquals(
        output_path="text",
        expected_path="text",
        fuzzy_enabled=True,
        fuzzy_threshold=0.8,
        normalize_lowercase=True,
        evaluation_name="fuzzy_match",
    )

    # Exact evaluator
    exact_eval = ScalarEquals(
        output_path="text",
        expected_path="text",
        fuzzy_enabled=False,
        normalize_lowercase=True,
        evaluation_name="exact_match",
    )

    print("Comparison Results:")
    print("=" * 80)
    print(f"{'AI Output':<35} | {'Expected':<35} | {'Fuzzy':<6} | {'Exact':<6}")
    print("-" * 80)

    fuzzy_matches = 0
    exact_matches = 0

    for case in test_cases:
        ctx = EvaluatorContext(
            name="test_context",
            inputs=None,
            metadata=None,
            output={"text": case["ai_output"]},
            expected_output={"text": case["expected"]},
            duration=0.0,
            _span_tree=None,
            attributes={},
            metrics={},
        )

        fuzzy_result = fuzzy_eval.evaluate(ctx)
        exact_result = exact_eval.evaluate(ctx)

        fuzzy_pass = "✅" if fuzzy_result.value >= 0.8 else "❌"
        exact_pass = "✅" if exact_result.value >= 0.8 else "❌"

        if fuzzy_result.value >= 0.8:
            fuzzy_matches += 1
        if exact_result.value >= 0.8:
            exact_matches += 1

        print(
            f"{case['ai_output'][:34]:<35} | {case['expected'][:34]:<35} | {fuzzy_pass:<6} | {exact_pass:<6}"
        )

    print("-" * 80)
    print(
        f"Total Matches: {' ' * 54} | {fuzzy_matches}/{len(test_cases):<6} | {exact_matches}/{len(test_cases):<6}"
    )
    print(
        f"Match Rate:    {' ' * 54} | {fuzzy_matches/len(test_cases):.1%} | {exact_matches/len(test_cases):.1%}"
    )


if __name__ == "__main__":
    run_content_evaluation_demo()
    compare_fuzzy_vs_exact()

    print("\n✨ Pipeline demo complete!")
    print("\nKey takeaways:")
    print("• Fuzzy matching significantly improves evaluation robustness")
    print("• Different thresholds work better for different content types")
    print("• Normalization is crucial for fair comparisons")
    print("• List evaluations benefit greatly from fuzzy matching")
