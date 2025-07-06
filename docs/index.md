# pydantic-ai-helpers

> Boring, opinionated helpers for PydanticAI that are so dumb you didn't want to implement them. So I did.

**⚠️ This is NOT an official PydanticAI package** - just a simple personal helper library.

[![PyPI version](https://img.shields.io/pypi/v/pydantic-ai-helpers.svg)](https://pypi.org/project/pydantic-ai-helpers/)
[![Python versions](https://img.shields.io/pypi/pyversions/pydantic-ai-helpers.svg)](https://pypi.org/project/pydantic-ai-helpers/)
[![CI status](https://github.com/yourusername/pydantic-ai-helpers/workflows/CI/badge.svg)](https://github.com/yourusername/pydantic-ai-helpers/actions)

## The Problem

[PydanticAI](https://github.com/pydantic/pydantic-ai) is amazing! But at some point you'll need to quickly and easily extract aspects of your conversations. It's not hard but it's a pain to do, because neither you nor the LLMS know how to do it, so you'll waste 10+ minutes to do:

```python
# Want the last tool call for your UI updates?
last_tool_call = None
for message in result.all_messages():
    for part in message.parts:
        if isinstance(part, ToolCallPart):
            last_tool_call = part

# Need that metadata you passed for evaluations?
metadata_parts = []
for message in result.all_messages():
    for part in message.parts:
        if isinstance(part, ToolReturnPart) and part.metadata:
            metadata_parts.append(part.metadata)

# How about just the user's question again?
user_question = None
for message in result.all_messages():
    for part in message.parts:
        if isinstance(part, UserPromptPart):
            user_question = part.content
            break
```

We've all been there. **We've got you!**

```python
from pydantic_ai_helpers import History
# or for convenience:
import pydantic_ai_helpers as ph

hist = History(result)  # or ph.History(result)
last_tool_call = hist.tools.calls().last()      # Done
metadata = hist.tools.returns().last().metadata  # Easy
user_question = hist.user.last().content        # Simple
system_prompt = hist.system_prompt()            # Get system message
media_items = hist.media.images()               # Extract media content
```

The best part? Your IDE will help you with the suggestions for the available methods so you don't have to remember anything!

## Installation

```bash
pip install pydantic-ai-helpers
```

Or with your favorite package manager:

```bash
poetry add pydantic-ai-helpers
uv add pydantic-ai-helpers
```

## Quick Start

```python
from pydantic_ai import Agent
from pydantic_ai_helpers import History
# or: import pydantic_ai_helpers as ph

agent = Agent("openai:gpt-4.1")
result = agent.run_sync("Tell me a joke")

# Wrap once, access everything
hist = History(result)  # or ph.History(result)

# Get the last user message
print(hist.user.last().content)
# Output: "Tell me a joke"

# Get all AI responses
for response in hist.ai.all():
    print(response.content)
    
# Check token usage
print(f"Tokens used: {hist.usage().total_tokens}")

# Access system prompt (if any)
if system_prompt := hist.system_prompt():
    print(f"System prompt: {system_prompt.content}")

# Access media content
images = hist.media.images()
if images:
    print(f"Found {len(images)} images in conversation")
```

## Examples

### Basic Usage

```python
from pydantic_ai import Agent
from pydantic_ai_helpers import History

def simple_conversation():
    """Basic conversation example."""
    agent = Agent("openai:gpt-4.1-mini", system_prompt="You are a helpful assistant.")

    # Run a simple query
    result = agent.run_sync("What is the capital of France?")

    # Wrap with History
    hist = History(result)

    # Access messages
    print(f"User asked: {hist.user.last().content}")
    print(f"AI responded: {hist.ai.last().content}")
    print(f"Tokens used: {hist.usage().total_tokens}")
```

### Multi-turn Conversations

```python
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
```

### Tool Usage Analysis

```python
from pydantic_ai import Tool

def tool_usage_example():
    """Example with tool usage."""
    # Define a simple tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
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
```

### Working with Media Content

```python
def media_analysis_example():
    """Example showing media content extraction."""
    # Assuming you have a conversation with media content
    hist = History(result)
    
    # Access all media content
    all_media = hist.media.all()
    print(f"Found {len(all_media)} media items")

    # Get specific media types
    images = hist.media.images()          # All images (URLs + binary)
    audio = hist.media.audio()            # All audio files
    documents = hist.media.documents()    # All documents
    videos = hist.media.videos()          # All videos

    # Filter by storage type
    url_images = hist.media.images(url_only=True)     # Only ImageUrl objects
    binary_images = hist.media.images(binary_only=True) # Only binary images

    # Get the most recent media
    latest_media = hist.media.last()
    if latest_media:
        print(f"Latest media: {type(latest_media).__name__}")

    # Filter by exact type
    from pydantic_ai.messages import ImageUrl, BinaryContent
    image_urls = hist.media.by_type(ImageUrl)
    binary_content = hist.media.by_type(BinaryContent)
```

### Streaming Support

```python
async def streaming_example():
    """Example with streaming responses."""
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
```

### Advanced Patterns

#### Conversation Persistence

```python
def conversation_persistence():
    """Save and restore conversation state."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter
    from pydantic_core import to_jsonable_python
    import json

    agent = Agent("openai:gpt-4o-mini")

    # Initial conversation
    result = agent.run_sync("Remember that my favorite color is blue")
    hist = History(result)

    # Save conversation state
    saved_messages = to_jsonable_python(hist.all_messages())
    with open("conversation_state.json", "w") as f:
        json.dump(saved_messages, f)

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
```

#### Cost Tracking

```python
def cost_tracking():
    """Track and estimate API costs."""
    # Approximate costs per 1K tokens (example rates)
    COSTS_PER_1K = {
        "gpt-4.1": {"input": 0.005, "output": 0.015},
        "gpt-4.1-mini": {"input": 0.00015, "output": 0.0006},
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
```

## API Reference

### `History` Class

The main wrapper class that provides access to all functionality.

**Constructor:**
- `History(result_or_messages)` - Accepts a `RunResult`, `StreamedRunResult`, or `list[ModelMessage]`

**Attributes:**
- `user: RoleView` - Access user messages
- `ai: RoleView` - Access AI messages  
- `system: RoleView` - Access system messages
- `tools: ToolsView` - Access tool calls and returns
- `media: MediaView` - Access media content in user messages

**Methods:**
- `all_messages() -> list[ModelMessage]` - Get raw message list
- `usage() -> Usage` - Aggregate token usage
- `tokens() -> Usage` - Alias for `usage()`
- `system_prompt() -> SystemPromptPart | None` - Get the first system prompt

### `RoleView` Class

Provides filtered access to messages by role.

**Methods:**
- `all() -> list[Part]` - Get all parts for this role
- `last() -> Part | None` - Get the most recent part

### `ToolsView` Class

Access tool-related messages.

**Methods:**
- `calls(*, name: str | None = None) -> ToolPartView` - Access tool calls
- `returns(*, name: str | None = None) -> ToolPartView` - Access tool returns

### `ToolPartView` Class

Filtered view of tool calls or returns.

**Methods:**
- `all() -> list[ToolCallPart | ToolReturnPart]` - Get all matching parts
- `last() -> ToolCallPart | ToolReturnPart | None` - Get the most recent part

### `MediaView` Class

Access media content from user messages (images, audio, documents, videos).

**Methods:**
- `all() -> list[MediaContent]` - Get all media content
- `last() -> MediaContent | None` - Get the most recent media item
- `images(*, url_only=False, binary_only=False)` - Get image content
- `audio(*, url_only=False, binary_only=False)` - Get audio content  
- `documents(*, url_only=False, binary_only=False)` - Get document content
- `videos(*, url_only=False, binary_only=False)` - Get video content
- `by_type(media_type)` - Get content by specific type (e.g., `ImageUrl`, `BinaryContent`)

## Common Patterns

### Check if a Tool Was Used

```python
if hist.tools.calls(name="calculator").last():
    result = hist.tools.returns(name="calculator").last()
    print(f"Calculation result: {result.content}")
```

### Count Message Types

```python
print(f"User messages: {len(hist.user.all())}")
print(f"AI responses: {len(hist.ai.all())}")
print(f"Tool calls: {len(hist.tools.calls().all())}")
print(f"Tool returns: {len(hist.tools.returns().all())}")
```

### Extract Conversation Text

```python
# Get all user inputs
user_inputs = [msg.content for msg in hist.user.all()]

# Get all AI responses  
ai_responses = [msg.content for msg in hist.ai.all()]

# Create a simple transcript
for user, ai in zip(user_inputs, ai_responses):
    print(f"User: {user}")
    print(f"AI: {ai}")
    print()
```

## Design Philosophy

1. **Boring is Good** - No clever magic, just simple method calls
2. **Autocomplete-Friendly** - Your IDE knows exactly what's available
3. **Zero Config** - Works out of the box with any PydanticAI result
4. **Type Safe** - Full type hints for everything
5. **Immutable** - History objects don't modify your data

## Contributing

Found a bug? Want a feature? PRs welcome!

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests (we maintain 100% coverage)
4. Make your changes
5. Run `make lint test` 
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

MIT - see [LICENSE](LICENSE) file.

---

Built with boredom-driven development. Because sometimes the most useful code is the code that does the obvious thing, obviously.