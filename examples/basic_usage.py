"""Basic usage examples for pydantic-ai-utils."""

import asyncio

from pydantic_ai import Agent, Tool

from pydantic_ai_helpers import History


def simple_conversation():
    """Run a basic conversation example."""
    agent = Agent("openai:gpt-4.1-mini", system_prompt="You are a helpful assistant.")

    # Run a simple query
    result = agent.run_sync("What is the capital of France?")

    # Wrap with History
    hist = History(result)

    # Access messages
    print(f"User asked: {hist.user.last().content}")
    print(f"AI responded: {hist.ai.last().content}")
    print(f"Tokens used: {hist.usage().total_tokens}")


def multi_turn_conversation():
    """Multi-turn conversation example."""
    agent = Agent("openai:gpt-4.1-mini")

    # Start conversation
    result = agent.run_sync("My name is Alice")
    hist = History(result)

    # Continue conversation
    result = agent.run_sync("What's my name?", message_history=hist.all_messages())
    hist = History(result)

    # Analyze the conversation
    print(f"Total exchanges: {len(hist.user.all())}")
    print("Conversation flow:")
    for i, (user, ai) in enumerate(zip(hist.user.all(), hist.ai.all(), strict=False)):
        print(f"  Turn {i + 1}:")
        print(f"    User: {user.content}")
        print(f"    AI: {ai.content}")


def tool_usage_example():
    """Run an example with tool usage."""

    # Define a simple tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        # Mock weather data
        weather_data = {
            "London": "Cloudy, 15°C",
            "Paris": "Sunny, 22°C",
            "Tokyo": "Rainy, 18°C",
        }
        return weather_data.get(city, "Unknown city")

    # Create agent with tool
    agent = Agent("openai:gpt-4.1-mini", tools=[Tool(get_weather)])

    # Run query that uses tool
    result = agent.run_sync("What's the weather in London and Paris?")
    hist = History(result)

    # Analyze tool usage
    print(f"Tool calls made: {len(hist.tools.calls().all())}")

    for call in hist.tools.calls().all():
        print(f"  Called {call.tool_name} with args: {call.args}")

    for ret in hist.tools.returns().all():
        print(f"  {ret.tool_name} returned: {ret.content}")

    print(f"\nFinal response: {hist.ai.last().content}")


def streaming_example():
    """Run an example with streaming responses."""

    async def stream_story():
        agent = Agent("openai:gpt-4.1-mini")

        async with agent.run_stream("Tell me a very short story") as result:
            print("Streaming: ", end="")
            async for chunk in result.stream():
                print(chunk, end="", flush=True)
            print()  # newline

            # After streaming, analyze with History
            hist = History(result)
            print(f"\nTotal tokens: {hist.usage().total_tokens}")
            print(f"Response tokens: {hist.usage().response_tokens}")

    # Run the async function
    asyncio.run(stream_story())


if __name__ == "__main__":
    print("=== Simple Conversation ===")
    simple_conversation()

    print("\n=== Multi-turn Conversation ===")
    multi_turn_conversation()

    print("\n=== Tool Usage ===")
    tool_usage_example()

    print("\n=== Streaming ===")
    streaming_example()
