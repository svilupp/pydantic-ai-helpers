"""Advanced usage patterns for pydantic-ai-utils."""

import json

from pydantic_ai import Agent, ModelRetry, Tool

from pydantic_ai_helpers import History


def conversation_analysis():
    """Analyze conversation patterns and metrics."""
    agent = Agent("openai:gpt-4.1-mini")

    # Simulate a longer conversation
    messages = []
    topics = [
        "What is machine learning?",
        "Can you give me an example?",
        "How does neural networks work?",
        "What about deep learning?",
    ]

    for topic in topics:
        result = agent.run_sync(topic, message_history=messages)
        messages = result.all_messages()

    hist = History(messages)

    # Conversation metrics
    print("Conversation Analysis:")
    print(f"  Total exchanges: {len(hist.user.all())}")
    print(
        f"  Average user message length: "
        f"{sum(len(m.content) for m in hist.user.all()) / len(hist.user.all()):.1f}"
        f" chars"
    )
    print(
        f"  Average AI response length: "
        f"{sum(len(m.content) for m in hist.ai.all()) / len(hist.ai.all()):.1f} chars"
    )
    print(f"  Total tokens used: {hist.usage().total_tokens}")
    print(
        f"  Tokens per exchange: {hist.usage().total_tokens / len(hist.user.all()):.1f}"
    )


def tool_retry_pattern():
    """Handle tool failures and retries."""
    call_count = 0

    def flaky_tool(number: int) -> str:
        """Return a doubled number, sometimes fails."""
        nonlocal call_count
        call_count += 1

        min_calls = 2
        if call_count < min_calls:
            raise ModelRetry("Service temporarily unavailable, please retry")

        return f"Success! The number is {number * 2}"

    agent = Agent("openai:gpt-4.1-mini", tools=[Tool(flaky_tool)], retries=3)

    result = agent.run_sync("Call the flaky tool with number 42")
    hist = History(result)

    # Analyze retry behavior
    tool_calls = hist.tools.calls(name="flaky_tool").all()
    tool_returns = hist.tools.returns(name="flaky_tool").all()

    print(f"Tool was called {len(tool_calls)} times")
    print(
        f"Successful returns: "
        f"{len([r for r in tool_returns if 'Success' in r.content])}"
    )
    print(f"Final result: {hist.ai.last().content}")


def conversation_persistence():
    """Save and restore conversation state."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter
    from pydantic_core import to_jsonable_python

    agent = Agent("openai:gpt-4o-mini")

    # Initial conversation
    result = agent.run_sync("Remember that my favorite color is blue")
    hist = History(result)

    # Save conversation state
    saved_messages = to_jsonable_python(hist.all_messages())
    with open("conversation_state.json", "w") as f:
        json.dump(saved_messages, f)

    print("Conversation saved.")

    # ... Later, in a new session ...

    # Load conversation state
    with open("conversation_state.json") as f:
        loaded_data = json.load(f)

    restored_messages = ModelMessagesTypeAdapter.validate_python(loaded_data)

    # Continue conversation
    result = agent.run_sync(
        "What's my favorite color?", message_history=restored_messages
    )

    hist = History(result)
    print(f"AI remembers: {hist.ai.last().content}")


def cost_tracking():
    """Track and estimate API costs."""
    # Approximate costs per 1K tokens (example rates)
    COSTS_PER_1K = {
        "gpt-4.1": {"input": 0.005, "output": 0.015},
        "gpt-4.1-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    model = "gpt-4.1-mini"
    agent = Agent(f"openai:{model}")

    # Run some queries
    queries = [
        "Explain quantum computing in one sentence",
        "Now explain it like I'm five",
        "What are practical applications?",
    ]

    total_cost = 0.0
    messages = []

    for query in queries:
        result = agent.run_sync(query, message_history=messages)
        messages = result.all_messages()
        hist = History(result)

        # Calculate cost for this exchange
        usage = hist.usage()
        if usage.request_tokens and usage.response_tokens:
            input_cost = (usage.request_tokens / 1000) * COSTS_PER_1K[model]["input"]
            output_cost = (usage.response_tokens / 1000) * COSTS_PER_1K[model]["output"]
            query_cost = input_cost + output_cost
            total_cost += query_cost

            print(f"Query: '{query[:30]}...'")
            print(f"  Tokens: {usage.request_tokens} in, {usage.response_tokens} out")
            print(f"  Cost: ${query_cost:.4f}")

    print(f"\nTotal cost for conversation: ${total_cost:.4f}")


def parallel_tool_execution():
    """Analyze parallel tool execution patterns."""
    import asyncio
    import time

    async def slow_tool_a(x: int) -> str:
        await asyncio.sleep(1)  # Simulate slow API
        return f"Tool A result: {x * 2}"

    async def slow_tool_b(x: int) -> str:
        await asyncio.sleep(1)  # Simulate slow API
        return f"Tool B result: {x * 3}"

    agent = Agent("openai:gpt-4o-mini", tools=[Tool(slow_tool_a), Tool(slow_tool_b)])

    async def analyze_parallel_execution():
        start_time = time.time()

        # This should trigger parallel tool execution
        result = await agent.run(
            "Call both slow_tool_a and slow_tool_b with the number 10"
        )

        elapsed = time.time() - start_time
        hist = History(result)

        # Analyze execution
        tool_calls = hist.tools.calls().all()
        print(f"Executed {len(tool_calls)} tools in {elapsed:.2f} seconds")

        parallel_threshold = 1.5
        if elapsed < parallel_threshold:  # Should be ~1 second if parallel
            print("✓ Tools executed in parallel")
        else:
            print("✗ Tools executed sequentially")

        for call in tool_calls:
            return_val = hist.tools.returns(name=call.tool_name).last()
            print(
                f"  {call.tool_name}: "
                f"{return_val.content if return_val else 'No return'}"
            )

    asyncio.run(analyze_parallel_execution())


def conversation_summarization():
    """Create summaries of long conversations."""
    agent = Agent("openai:gpt-4o-mini")

    # Simulate a long conversation
    conversation_topics = [
        "Tell me about the Python programming language",
        "What are Python's main use cases?",
        "How does Python compare to JavaScript?",
        "What frameworks are popular in Python?",
        "Can you show me a simple Python web server?",
    ]

    messages = []
    for topic in conversation_topics:
        result = agent.run_sync(topic, message_history=messages)
        messages = result.all_messages()

    hist = History(messages)

    # Create a conversation summary
    print("Conversation Summary:")
    print(f"Topics discussed: {len(hist.user.all())}")
    print(f"Total tokens: {hist.usage().total_tokens}")
    print("\nUser questions:")
    for i, q in enumerate(hist.user.all(), 1):
        print(f"  {i}. {q.content[:50]}...")

    # Use another agent to summarize the conversation
    summarizer = Agent(
        "openai:gpt-4.1-mini",
        system_prompt="You are a conversation summarizer. Create concise summaries.",
    )

    # Prepare conversation text
    conversation_text = "\n".join(
        [
            f"User: {user.content}\nAssistant: {ai.content}"
            for user, ai in zip(hist.user.all(), hist.ai.all(), strict=False)
        ]
    )

    summary_result = summarizer.run_sync(
        f"Summarize this conversation in 3 bullet points:\n\n"
        f"{conversation_text[:2000]}..."
    )

    summary_hist = History(summary_result)
    print(f"\nAI Summary:\n{summary_hist.ai.last().content}")


if __name__ == "__main__":
    print("=== Conversation Analysis ===")
    conversation_analysis()

    print("\n=== Tool Retry Pattern ===")
    tool_retry_pattern()

    print("\n=== Conversation Persistence ===")
    conversation_persistence()

    print("\n=== Cost Tracking ===")
    cost_tracking()

    print("\n=== Parallel Tool Execution ===")
    parallel_tool_execution()

    print("\n=== Conversation Summarization ===")
    conversation_summarization()
