#!/usr/bin/env python3
"""
Generate and serialize test conversations using PydanticAI.
Creates various conversation patterns including multi-turn dialogues and tool usage.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel
from pydantic_core import to_jsonable_python
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessagesTypeAdapter


# Create output directory for serialized conversations
output_dir = Path("test_conversations")
output_dir.mkdir(exist_ok=True)


def save_conversation(name: str, messages: List[Any]) -> None:
    """Save a conversation to a JSON file."""
    filename = output_dir / f"{name}.json"
    serialized = to_jsonable_python(messages)
    with open(filename, "w") as f:
        json.dump(serialized, f, indent=2)
    print(f"Saved conversation: {filename}")


# Example 1: Simple multi-turn conversation
def generate_simple_conversation():
    """Generate a basic multi-turn conversation."""
    agent = Agent("openai:gpt-4o", system_prompt="Be a helpful assistant.")
    
    # First turn
    result1 = agent.run_sync("Tell me a joke about programming.")
    
    # Second turn using history
    result2 = agent.run_sync(
        "Can you explain why that's funny?", 
        message_history=result1.all_messages()
    )
    
    # Third turn
    result3 = agent.run_sync(
        "Tell me another programming joke.", 
        message_history=result2.all_messages()
    )
    
    save_conversation("simple_multiturn", result3.all_messages())


# Example 2: Conversation with tool usage (dice game)
def generate_tool_conversation():
    """Generate a conversation using tools."""
    agent = Agent(
        "openai:gpt-4o",
        deps_type=str,
        system_prompt=(
            "You're a dice game host. Roll the die and see if the number "
            "matches the user's guess. Use the player's name in responses."
        ),
    )
    
    @agent.tool_plain
    def roll_dice() -> str:
        """Roll a six-sided die and return the result."""
        return str(random.randint(1, 6))
    
    @agent.tool
    def get_player_name(ctx: RunContext[str]) -> str:
        """Get the player's name."""
        return ctx.deps
    
    # Multiple rounds of the game
    result1 = agent.run_sync("My guess is 4", deps="Alice")
    result2 = agent.run_sync(
        "Let me try again, I guess 2", 
        deps="Alice",
        message_history=result1.all_messages()
    )
    result3 = agent.run_sync(
        "One more time! I pick 6", 
        deps="Alice",
        message_history=result2.all_messages()
    )
    
    save_conversation("dice_game_with_tools", result3.all_messages())


# Example 3: Math tutor with structured output
class MathProblem(BaseModel):
    problem: str
    solution: str
    explanation: str


def generate_structured_conversation():
    """Generate a conversation with structured outputs."""
    agent = Agent(
        "openai:gpt-4o",
        result_type=MathProblem,
        system_prompt="You are a math tutor. Solve problems step by step."
    )
    
    result1 = agent.run_sync("What is 15% of 80?")
    
    # Follow-up with a harder problem
    result2 = agent.run_sync(
        "Now solve: If a shirt costs $40 after a 20% discount, what was the original price?",
        message_history=result1.all_messages()
    )
    
    save_conversation("math_tutor_structured", result2.all_messages())


# Example 4: Creative writing assistant
def generate_creative_conversation():
    """Generate a creative writing conversation."""
    agent = Agent(
        "openai:gpt-4o",
        system_prompt="You are a creative writing assistant helping with story development."
    )
    
    result1 = agent.run_sync("Help me create a character for a sci-fi story.")
    result2 = agent.run_sync(
        "Great! Now what kind of conflict should this character face?",
        message_history=result1.all_messages()
    )
    result3 = agent.run_sync(
        "How would this character's background influence their approach to this conflict?",
        message_history=result2.all_messages()
    )
    result4 = agent.run_sync(
        "Write a short opening paragraph for this story.",
        message_history=result3.all_messages()
    )
    
    save_conversation("creative_writing_4turns", result4.all_messages())


# Example 5: Code review assistant with tools
def generate_code_review_conversation():
    """Generate a code review conversation with analysis tools."""
    agent = Agent(
        "openai:gpt-4o",
        system_prompt="You are a code review assistant. Analyze code and suggest improvements."
    )
    
    @agent.tool_plain
    def analyze_complexity(code: str) -> str:
        """Analyze code complexity (mock implementation)."""
        lines = code.strip().split('\n')
        return f"Code has {len(lines)} lines with moderate complexity"
    
    @agent.tool_plain
    def check_style_issues(code: str) -> List[str]:
        """Check for style issues (mock implementation)."""
        issues = []
        if "TODO" in code:
            issues.append("Contains TODO comments")
        if not code.strip().startswith('"""'):
            issues.append("Missing module docstring")
        return issues if issues else ["No style issues found"]
    
    code_sample = '''
def calculate_average(numbers):
    # TODO: Add error handling
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
'''
    
    result1 = agent.run_sync(f"Review this Python function:\n```python\n{code_sample}\n```")
    result2 = agent.run_sync(
        "What specific improvements would you suggest for error handling?",
        message_history=result1.all_messages()
    )
    result3 = agent.run_sync(
        "Show me the improved version of the function.",
        message_history=result2.all_messages()
    )
    
    save_conversation("code_review_with_tools", result3.all_messages())


# Example 6: Single turn conversation
def generate_single_turn():
    """Generate a single-turn conversation."""
    agent = Agent("openai:gpt-4o", system_prompt="You are a helpful assistant.")
    result = agent.run_sync("What's the capital of France?")
    save_conversation("single_turn_simple", result.all_messages())


# Example 7: Conversation with context switching
def generate_context_switch():
    """Generate a conversation that switches topics."""
    agent = Agent("openai:gpt-4o", system_prompt="You are a versatile assistant.")
    
    result1 = agent.run_sync("Explain quantum computing in simple terms.")
    result2 = agent.run_sync(
        "Now let's switch topics. What are the health benefits of meditation?",
        message_history=result1.all_messages()
    )
    result3 = agent.run_sync(
        "Going back to quantum computing, what are its practical applications?",
        message_history=result2.all_messages()
    )
    
    save_conversation("context_switching", result3.all_messages())


def main():
    """Run all conversation generators."""
    print("Generating test conversations...")
    
    generators = [
        ("Simple multi-turn", generate_simple_conversation),
        ("Dice game with tools", generate_tool_conversation),
        ("Structured math tutor", generate_structured_conversation),
        ("Creative writing", generate_creative_conversation),
        ("Code review with tools", generate_code_review_conversation),
        ("Single turn", generate_single_turn),
        ("Context switching", generate_context_switch),
    ]
    
    for name, generator in generators:
        try:
            print(f"\nGenerating: {name}")
            generator()
        except Exception as e:
            print(f"Error generating {name}: {e}")
    
    print(f"\nAll conversations saved to: {output_dir}")
    
    # Also demonstrate loading a conversation back
    print("\nDemonstrating conversation loading...")
    with open(output_dir / "simple_multiturn.json") as f:
        loaded_data = json.load(f)
    
    # Validate and reconstruct the messages
    reconstructed = ModelMessagesTypeAdapter.validate_python(loaded_data)
    print(f"Successfully loaded conversation with {len(reconstructed)} messages")


if __name__ == "__main__":
    main()