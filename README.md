# pydantic-ai-helpers

> Boring, opinionated helpers for PydanticAI that are so dumb you didn't want to implement them. So I did.

**⚠️ This is NOT an official PydanticAI package** - just a simple personal helper library.

[![PyPI version](https://img.shields.io/pypi/v/pydantic-ai-helpers.svg)](https://pypi.org/project/pydantic-ai-helpers/)
[![Python versions](https://img.shields.io/pypi/pyversions/pydantic-ai-helpers.svg)](https://pypi.org/project/pydantic-ai-helpers/)
[![CI status](https://github.com/yourusername/pydantic-ai-helpers/workflows/CI/badge.svg)](https://github.com/yourusername/pydantic-ai-helpers/actions)
[![Coverage](https://img.shields.io/codecov/c/github/yourusername/pydantic-ai-helpers)](https://codecov.io/gh/yourusername/pydantic-ai-helpers)
[![License](https://img.shields.io/pypi/l/pydantic-ai-helpers.svg)](https://github.com/yourusername/pydantic-ai-helpers/blob/main/LICENSE)

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
uv add pydantic-ai-helpers
# pip install pydantic-ai-helpers
# poetry add pydantic-ai-helpers
```

## Quick Start

```python
from pydantic_ai import Agent
from pydantic_ai_helpers import History
# or: import pydantic_ai_helpers as ph

agent = Agent("openai:gpt-4.1-mini")
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

## Common Use Cases

### Extract What You Need for Your App

```python
hist = History(result)

# Update your UI with the latest tool status
if latest_call := hist.tools.calls().last():
    update_ui_status(f"Called {latest_call.tool_name}...")

# Get conversation context for logging
user_query = hist.user.last().content
ai_response = hist.ai.last().content
log_conversation(user_query, ai_response)

# Check token costs for billing
total_cost = hist.usage().total_tokens * your_token_rate
```

### Debug Tool Workflows

```python
# See what tools were actually called
for call in hist.tools.calls().all():
    print(f"Called {call.tool_name} with {call.args}")

# Check what came back
for ret in hist.tools.returns().all():
    print(f"{ret.tool_name} returned: {ret.content}")
    if ret.metadata:  # Your evaluation metadata
        print(f"Metadata: {ret.metadata}")
```

### Analyze Conversations

```python
# Count interactions
print(f"User asked {len(hist.user.all())} questions")
print(f"AI made {len(hist.tools.calls().all())} tool calls")
print(f"Total tokens: {hist.usage().total_tokens}")

# Get specific tool results for processing
weather_results = hist.tools.returns(name="get_weather").all()
for result in weather_results:
    process_weather_data(result.content)
```

### Work with Media Content

```python
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

### Access System Prompts

```python
# Get the system prompt (if any)
system_prompt = hist.system_prompt()
if system_prompt:
    print(f"System prompt: {system_prompt.content}")
else:
    print("No system prompt found")

# Use in analysis
if system_prompt and "helpful" in system_prompt.content:
    print("This agent was configured to be helpful")
```

## Examples

### Multi-turn Conversation Analysis

```python
messages = []
topics = [
    "What's the weather in London?",
    "How about Paris?",
    "Which city is warmer?"
]

for topic in topics:
    result = agent.run_sync(topic, message_history=messages)
    messages = result.all_messages()

hist = History(result)

# Analyze the conversation flow
print(f"User asked {len(hist.user.all())} questions")
print(f"AI responded {len(hist.ai.all())} times")
print(f"Made {len(hist.tools.calls())} tool calls")

# Get specific information
london_weather = hist.tools.returns(name="get_weather").all()[0]
paris_weather = hist.tools.returns(name="get_weather").all()[1]
```

### Dice Game with Tools

```python
# From the PydanticAI tutorial
result = agent.run_sync("Roll a dice")

hist = History(result)

# Find what the dice rolled
dice_result = hist.tools.returns(name="roll_dice").last()
print(f"Dice rolled: {dice_result.content}")

# See how the AI responded
ai_message = hist.ai.last()
print(f"AI said: {ai_message.content}")
```

### Streaming Support

```python
async with agent.run_stream("Tell me a story") as result:
    async for chunk in result.stream():
        print(chunk, end="")
    
    # After streaming completes
    hist = History(result)
    print(f"\nTotal tokens: {hist.tokens().total_tokens}")
```

### Loading from Serialized Conversations

```python
import json
from pydantic_core import to_jsonable_python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter  

# Save a conversation
agent = Agent('openai:gpt-4.1-mini')
result = agent.run_sync('Tell me a joke.')
messages = result.all_messages()

# Serialize to file
with open('conversation.json', 'w') as f:
    json.dump(to_jsonable_python(messages), f)

# Later, load it back
hist = History('conversation.json')
print(hist)  # History(1 turn, 50 tokens)
print(hist.user.last().content)  # "Tell me a joke."
print(hist.ai.last().content)    # The joke response

# Or use Path objects
from pathlib import Path
hist = History(Path('conversation.json'))

# Continue the conversation with loaded history
same_messages = ModelMessagesTypeAdapter.validate_python(
    to_jsonable_python(hist.all_messages())
)
result2 = agent.run_sync(
    'Tell me a different joke.', 
    message_history=same_messages
)
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

### Work with Media Content

```python
# Check if conversation has images
if hist.media.images():
    print("This conversation contains images")
    for img in hist.media.images():
        if hasattr(img, 'url'):
            print(f"Image URL: {img.url}")
        else:
            print(f"Binary image: {img.media_type}, {len(img.data)} bytes")

# Process different media types
for media_item in hist.media.all():
    if isinstance(media_item, ImageUrl):
        download_image(media_item.url)
    elif isinstance(media_item, BinaryContent):
        save_binary_content(media_item.data, media_item.media_type)
```

### Extract System Configuration

```python
# Check system prompt for agent behavior
system_prompt = hist.system_prompt()
if system_prompt:
    if "helpful" in system_prompt.content.lower():
        agent_type = "helpful_assistant"
    elif "creative" in system_prompt.content.lower():
        agent_type = "creative_writer"
    else:
        agent_type = "general_purpose"
    
    print(f"Agent type: {agent_type}")
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

## Development

```bash
# Clone the repo
git clone https://github.com/yourusername/pydantic-ai-helpers.git
cd pydantic-ai-helpers

# Install in development mode
make install

# Run tests
make test

# Run linting
make lint

# Format code
make format
```

## License

MIT - see [LICENSE](LICENSE) file.

---

Built with boredom-driven development. Because sometimes the most useful code is the code that does the obvious thing, obviously.