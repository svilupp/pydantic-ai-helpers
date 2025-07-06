"""Comprehensive test suite for the History wrapper."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.usage import Usage
from pydantic_core import to_jsonable_python

import pydantic_ai_helpers as ph
from pydantic_ai_helpers import History


class TestHistoryInit:
    """Test History constructor with various input types."""

    def test_init_with_runresult(self) -> None:
        """Test initialization with a RunResult-like object."""
        mock_result = Mock()
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there!")]),
        ]
        mock_result.all_messages.return_value = messages

        hist = History(mock_result)
        assert hist.all_messages() == messages

    def test_init_with_message_list(self) -> None:
        """Test initialization with a list of ModelMessage."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there!")]),
        ]

        hist = History(messages)
        assert hist.all_messages() == messages

    def test_init_with_empty_list(self) -> None:
        """Test initialization with an empty list."""
        hist = History([])
        assert hist.all_messages() == []
        assert hist.user.all() == []
        assert hist.ai.all() == []

    def test_init_with_invalid_type(self) -> None:
        """Test initialization with an invalid type."""
        # Now strings are treated as file paths, so we get ValueError instead
        with pytest.raises(ValueError, match=r"does not exist"):
            History("not a valid input")

    def test_init_with_numeric_invalid_type(self) -> None:
        """Test initialization with a numeric type."""
        with pytest.raises(TypeError, match=r"History\(\) expects"):
            History(42)

    def test_init_with_non_iterable_invalid_type(self) -> None:
        """Test initialization with a non-iterable invalid type."""
        class NonIterable:
            pass

        with pytest.raises(TypeError, match=r"History\(\) expects"):
            History(NonIterable())

    def test_init_with_invalid_iterable(self) -> None:
        """Test initialization with a non-message iterable."""
        with pytest.raises(TypeError, match=r"History\(\) expects"):
            History([1, 2, 3])

    def test_init_with_mixed_invalid_iterable(self) -> None:
        """Test initialization with an iterable containing valid and invalid items."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            "not a message",  # Invalid item
        ]
        with pytest.raises(TypeError, match=r"History\(\) expects"):
            History(messages)

    def test_init_with_non_empty_invalid_iterable(self) -> None:
        """Test initialization with a non-empty iterable of invalid items."""
        # This should specifically test the branch that falls through to TypeError
        # when we have an iterable that's not empty but doesn't contain ModelMessages
        class FakeObject:
            pass

        fake_objects = [FakeObject(), FakeObject()]
        with pytest.raises(TypeError, match=r"History\(\) expects"):
            History(fake_objects)


class TestRoleViews:
    """Test role-based views (user, ai, system)."""

    @pytest.fixture
    def sample_messages(self) -> list[ModelMessage]:
        """Create a sample conversation with multiple message types."""
        return [
            ModelRequest(parts=[UserPromptPart(content="First user message")]),
            ModelResponse(parts=[TextPart(content="First AI response")]),
            ModelRequest(parts=[UserPromptPart(content="Second user message")]),
            ModelResponse(parts=[TextPart(content="Second AI response")]),
            ModelRequest(parts=[SystemPromptPart(content="System prompt")]),
        ]

    def test_user_view_all(self, sample_messages: list[ModelMessage]) -> None:
        """Test getting all user messages."""
        hist = History(sample_messages)
        user_parts = hist.user.all()

        assert len(user_parts) == 2
        assert all(isinstance(p, UserPromptPart) for p in user_parts)
        assert user_parts[0].content == "First user message"
        assert user_parts[1].content == "Second user message"

    def test_user_view_last(self, sample_messages: list[ModelMessage]) -> None:
        """Test getting the last user message."""
        hist = History(sample_messages)
        last_user = hist.user.last()

        assert isinstance(last_user, UserPromptPart)
        assert last_user.content == "Second user message"

    def test_ai_view_all(self, sample_messages: list[ModelMessage]) -> None:
        """Test getting all AI messages."""
        hist = History(sample_messages)
        ai_parts = hist.ai.all()

        assert len(ai_parts) == 2
        assert all(isinstance(p, TextPart) for p in ai_parts)
        assert ai_parts[0].content == "First AI response"
        assert ai_parts[1].content == "Second AI response"

    def test_ai_view_last(self, sample_messages: list[ModelMessage]) -> None:
        """Test getting the last AI message."""
        hist = History(sample_messages)
        last_ai = hist.ai.last()

        assert isinstance(last_ai, TextPart)
        assert last_ai.content == "Second AI response"

    def test_system_view(self, sample_messages: list[ModelMessage]) -> None:
        """Test system message access."""
        hist = History(sample_messages)
        system_parts = hist.system.all()

        assert len(system_parts) == 1
        assert isinstance(system_parts[0], SystemPromptPart)
        assert system_parts[0].content == "System prompt"

    def test_empty_role_last_returns_none(self) -> None:
        """Test that last() returns None for empty views."""
        hist = History([])
        assert hist.user.last() is None
        assert hist.ai.last() is None
        assert hist.system.last() is None


class TestToolViews:
    """Test tool-related views."""

    @pytest.fixture
    def tool_messages(self) -> list[ModelMessage]:
        """Create messages with tool calls and returns."""
        return [
            ModelRequest(parts=[UserPromptPart(content="Roll a dice")]),
            ModelResponse(parts=[
                ToolCallPart(
                    tool_name="roll_dice",
                    args={"sides": 6},
                    tool_call_id="call_123"
                )
            ]),
            ModelRequest(parts=[
                ToolReturnPart(
                    tool_name="roll_dice",
                    content="4",
                    tool_call_id="call_123"
                )
            ]),
            ModelResponse(parts=[
                TextPart(content="You rolled a 4!"),
                ToolCallPart(
                    tool_name="get_weather",
                    args={"city": "London"},
                    tool_call_id="call_456"
                )
            ]),
            ModelRequest(parts=[
                ToolReturnPart(
                    tool_name="get_weather",
                    content="Rainy, 15°C",
                    tool_call_id="call_456"
                )
            ]),
        ]

    def test_tools_calls_all(self, tool_messages: list[ModelMessage]) -> None:
        """Test getting all tool calls."""
        hist = History(tool_messages)
        calls = hist.tools.calls().all()

        assert len(calls) == 2
        assert all(isinstance(c, ToolCallPart) for c in calls)
        assert calls[0].tool_name == "roll_dice"
        assert calls[1].tool_name == "get_weather"

    def test_tools_calls_filtered(self, tool_messages: list[ModelMessage]) -> None:
        """Test filtering tool calls by name."""
        hist = History(tool_messages)
        dice_calls = hist.tools.calls(name="roll_dice").all()

        assert len(dice_calls) == 1
        assert dice_calls[0].tool_name == "roll_dice"
        # Type narrowing for mypy
        assert isinstance(dice_calls[0], ToolCallPart)
        assert dice_calls[0].args == {"sides": 6}

    def test_tools_returns_all(self, tool_messages: list[ModelMessage]) -> None:
        """Test getting all tool returns."""
        hist = History(tool_messages)
        returns = hist.tools.returns().all()

        assert len(returns) == 2
        assert all(isinstance(r, ToolReturnPart) for r in returns)
        # Type narrowing for mypy
        assert isinstance(returns[0], ToolReturnPart)
        assert isinstance(returns[1], ToolReturnPart)
        assert returns[0].content == "4"
        assert returns[1].content == "Rainy, 15°C"

    def test_tools_returns_filtered(self, tool_messages: list[ModelMessage]) -> None:
        """Test filtering tool returns by name."""
        hist = History(tool_messages)
        weather_return = hist.tools.returns(name="get_weather").last()

        assert weather_return is not None
        assert weather_return.tool_name == "get_weather"
        # Type narrowing for mypy
        assert isinstance(weather_return, ToolReturnPart)
        assert weather_return.content == "Rainy, 15°C"

    def test_tools_empty_filtered(self, tool_messages: list[ModelMessage]) -> None:
        """Test filtering with non-existent tool name."""
        hist = History(tool_messages)

        assert hist.tools.calls(name="nonexistent").all() == []
        assert hist.tools.calls(name="nonexistent").last() is None

    def test_tool_return_metadata(self) -> None:
        """Test accessing tool return metadata."""
        messages = [
            ModelRequest(parts=[
                ToolReturnPart(
                    tool_name="complex_tool",
                    content="Result",
                    tool_call_id="call_789",
                    metadata={"execution_time": 1.23, "cache_hit": True}
                )
            ]),
        ]

        hist = History(messages)
        tool_return = hist.tools.returns().last()

        assert tool_return is not None
        # Type narrowing for mypy
        assert isinstance(tool_return, ToolReturnPart)
        assert tool_return.metadata == {"execution_time": 1.23, "cache_hit": True}


class TestUsageAggregation:
    """Test usage/token aggregation functionality."""

    def test_usage_single_response(self) -> None:
        """Test usage aggregation with a single response."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(
                parts=[TextPart(content="Hi!")],
                usage=Usage(
                    requests=1,
                    request_tokens=10,
                    response_tokens=5,
                    total_tokens=15
                )
            ),
        ]

        hist = History(messages)
        usage = hist.usage()

        assert usage.requests == 1
        assert usage.request_tokens == 10
        assert usage.response_tokens == 5
        assert usage.total_tokens == 15

    def test_usage_multiple_responses(self) -> None:
        """Test usage aggregation across multiple responses."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(
                parts=[TextPart(content="Hi!")],
                usage=Usage(
                    requests=1,
                    request_tokens=10,
                    response_tokens=5,
                    total_tokens=15
                )
            ),
            ModelRequest(parts=[UserPromptPart(content="How are you?")]),
            ModelResponse(
                parts=[TextPart(content="I'm great!")],
                usage=Usage(
                    requests=1,
                    request_tokens=20,
                    response_tokens=10,
                    total_tokens=30
                )
            ),
        ]

        hist = History(messages)
        usage = hist.usage()

        assert usage.requests == 2
        assert usage.request_tokens == 30
        assert usage.response_tokens == 15
        assert usage.total_tokens == 45

    def test_usage_with_details(self) -> None:
        """Test usage aggregation with details dict."""
        messages = [
            ModelResponse(
                parts=[TextPart(content="Response 1")],
                usage=Usage(
                    requests=1,
                    total_tokens=100,
                    details={"cache_tokens": 50, "reasoning_tokens": 10}
                )
            ),
            ModelResponse(
                parts=[TextPart(content="Response 2")],
                usage=Usage(
                    requests=1,
                    total_tokens=200,
                    details={"cache_tokens": 75, "reasoning_tokens": 25}
                )
            ),
        ]

        hist = History(messages)
        usage = hist.usage()

        assert usage.total_tokens == 300
        assert usage.details == {"cache_tokens": 125, "reasoning_tokens": 35}

    def test_usage_no_data(self) -> None:
        """Test usage with no usage data in messages."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi!")]),  # No usage data
        ]

        hist = History(messages)
        usage = hist.usage()

        assert usage.requests == 0
        assert usage.request_tokens == 0
        assert usage.response_tokens == 0
        assert usage.total_tokens == 0
        assert usage.details is None

    def test_tokens_alias(self) -> None:
        """Test that tokens() is an alias for usage()."""
        messages = [
            ModelResponse(
                parts=[TextPart(content="Hi!")],
                usage=Usage(total_tokens=42)
            ),
        ]

        hist = History(messages)
        assert hist.tokens().total_tokens == hist.usage().total_tokens


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_mixed_parts_in_single_message(self) -> None:
        """Test messages with multiple part types."""
        messages = [
            ModelResponse(parts=[
                TextPart(content="Here's the weather:"),
                ToolCallPart(
                    tool_name="get_weather",
                    args={},
                    tool_call_id="call_1"
                ),
                TextPart(content="Let me check for you."),
            ]),
        ]

        hist = History(messages)
        ai_parts = hist.ai.all()
        tool_calls = hist.tools.calls().all()

        assert len(ai_parts) == 2
        assert len(tool_calls) == 1

    def test_immutability(self) -> None:
        """Test that History doesn't modify original messages."""
        original_messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ]

        hist = History(original_messages)
        returned_messages = hist.all_messages()

        # Modifying returned list shouldn't affect History
        returned_messages.append(
            ModelResponse(parts=[TextPart(content="New message")])
        )

        assert len(hist.all_messages()) == 1
        assert len(original_messages) == 1

    def test_usage_with_none_fields(self) -> None:
        """Test usage aggregation with None fields."""
        messages = [
            ModelResponse(
                parts=[TextPart(content="Hi!")],
                usage=Usage(
                    requests=0,
                    request_tokens=0,
                    response_tokens=10,
                    total_tokens=10
                )
            ),
        ]

        hist = History(messages)
        usage = hist.usage()

        assert usage.requests == 0
        assert usage.request_tokens == 0
        assert usage.response_tokens == 10
        assert usage.total_tokens == 10


class TestSystemPrompt:
    """Test system prompt extraction."""

    def test_system_prompt_exists(self) -> None:
        """Test system_prompt() returns the first system prompt."""
        messages = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="You are a helpful assistant."),
                    UserPromptPart(content="Hello"),
                ]
            ),
            ModelResponse(parts=[TextPart(content="Hi there!")]),
        ]

        hist = History(messages)
        prompt = hist.system_prompt()

        assert prompt is not None
        assert isinstance(prompt, SystemPromptPart)
        assert prompt.content == "You are a helpful assistant."

    def test_system_prompt_none_when_absent(self) -> None:
        """Test system_prompt() returns None when no system prompt exists."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there!")]),
        ]

        hist = History(messages)
        prompt = hist.system_prompt()

        assert prompt is None

    def test_system_prompt_first_occurrence(self) -> None:
        """Test system_prompt() returns the first system prompt when multiple exist."""
        messages = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="First system prompt"),
                    UserPromptPart(content="Hello"),
                ]
            ),
            ModelResponse(parts=[TextPart(content="Hi there!")]),
            ModelRequest(
                parts=[
                    SystemPromptPart(content="Second system prompt"),
                    UserPromptPart(content="Another message"),
                ]
            ),
        ]

        hist = History(messages)
        prompt = hist.system_prompt()

        assert prompt is not None
        assert isinstance(prompt, SystemPromptPart)
        assert prompt.content == "First system prompt"

    def test_system_prompt_empty_history(self) -> None:
        """Test system_prompt() returns None for empty history."""
        hist = History([])
        prompt = hist.system_prompt()

        assert prompt is None


class TestMediaView:
    """Test media content extraction and filtering."""

    def test_media_empty_history(self) -> None:
        """Test media view with empty history."""
        hist = History([])
        assert hist.media.all() == []
        assert hist.media.last() is None
        assert hist.media.images() == []
        assert hist.media.audio() == []
        assert hist.media.documents() == []
        assert hist.media.videos() == []

    def test_media_text_only(self) -> None:
        """Test media view with text-only messages."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there!")]),
        ]

        hist = History(messages)
        assert hist.media.all() == []
        assert hist.media.last() is None
        assert hist.media.images() == []

    def test_media_single_image_url(self) -> None:
        """Test media view with single ImageUrl."""
        image_url = ImageUrl(url="https://example.com/image.jpg")
        messages = [
            ModelRequest(parts=[
                UserPromptPart(content=["Look at this:", image_url])
            ]),
            ModelResponse(parts=[TextPart(content="Nice image!")]),
        ]

        hist = History(messages)

        # Test all media
        all_media = hist.media.all()
        assert len(all_media) == 1
        assert all_media[0] == image_url

        # Test last media
        assert hist.media.last() == image_url

        # Test images
        images = hist.media.images()
        assert len(images) == 1
        assert images[0] == image_url

        # Test filtered images
        url_images = hist.media.images(url_only=True)
        assert len(url_images) == 1
        assert url_images[0] == image_url

        binary_images = hist.media.images(binary_only=True)
        assert len(binary_images) == 0

        # Test by_type
        image_urls = hist.media.by_type(ImageUrl)
        assert len(image_urls) == 1
        assert image_urls[0] == image_url

    def test_media_single_binary_content(self) -> None:
        """Test media view with single BinaryContent."""
        binary_content = BinaryContent(data=b"fake_image_data", media_type="image/jpeg")
        messages = [
            ModelRequest(parts=[
                UserPromptPart(content=["Here's an image:", binary_content])
            ]),
            ModelResponse(parts=[TextPart(content="Got it!")]),
        ]

        hist = History(messages)

        # Test all media
        all_media = hist.media.all()
        assert len(all_media) == 1
        assert all_media[0] == binary_content

        # Test images
        images = hist.media.images()
        assert len(images) == 1
        assert images[0] == binary_content

        # Test filtered images
        url_images = hist.media.images(url_only=True)
        assert len(url_images) == 0

        binary_images = hist.media.images(binary_only=True)
        assert len(binary_images) == 1
        assert binary_images[0] == binary_content

        # Test by_type
        binary_contents = hist.media.by_type(BinaryContent)
        assert len(binary_contents) == 1
        assert binary_contents[0] == binary_content

    def test_media_mixed_content(self) -> None:
        """Test media view with mixed media types."""
        image_url = ImageUrl(url="https://example.com/image.jpg")
        audio_url = AudioUrl(url="https://example.com/audio.mp3")
        doc_url = DocumentUrl(url="https://example.com/doc.pdf")
        video_url = VideoUrl(url="https://example.com/video.mp4")
        binary_image = BinaryContent(data=b"image_data", media_type="image/png")
        binary_audio = BinaryContent(data=b"audio_data", media_type="audio/wav")

        messages = [
            ModelRequest(parts=[
                UserPromptPart(content=[
                    "Check these out:",
                    image_url,
                    audio_url,
                    doc_url,
                    video_url,
                    binary_image,
                    binary_audio
                ])
            ]),
            ModelResponse(parts=[TextPart(content="All received!")]),
        ]

        hist = History(messages)

        # Test all media
        all_media = hist.media.all()
        assert len(all_media) == 6

        # Test last media
        assert hist.media.last() == binary_audio

        # Test images
        images = hist.media.images()
        assert len(images) == 2
        assert image_url in images
        assert binary_image in images

        # Test audio
        audio = hist.media.audio()
        assert len(audio) == 2
        assert audio_url in audio
        assert binary_audio in audio

        # Test documents
        documents = hist.media.documents()
        assert len(documents) == 1
        assert doc_url in documents

        # Test videos
        videos = hist.media.videos()
        assert len(videos) == 1
        assert video_url in videos

        # Test URL-only filtering
        url_images = hist.media.images(url_only=True)
        assert len(url_images) == 1
        assert url_images[0] == image_url

        url_audio = hist.media.audio(url_only=True)
        assert len(url_audio) == 1
        assert url_audio[0] == audio_url

        # Test binary-only filtering
        binary_images = hist.media.images(binary_only=True)
        assert len(binary_images) == 1
        assert binary_images[0] == binary_image

        binary_audio_items = hist.media.audio(binary_only=True)
        assert len(binary_audio_items) == 1
        assert binary_audio_items[0] == binary_audio

    def test_media_filtering_errors(self) -> None:
        """Test media filtering with conflicting parameters."""
        hist = History([])

        with pytest.raises(
            ValueError, match="Cannot specify both url_only and binary_only"
        ):
            hist.media.images(url_only=True, binary_only=True)

        with pytest.raises(
            ValueError, match="Cannot specify both url_only and binary_only"
        ):
            hist.media.audio(url_only=True, binary_only=True)

        with pytest.raises(
            ValueError, match="Cannot specify both url_only and binary_only"
        ):
            hist.media.documents(url_only=True, binary_only=True)

        with pytest.raises(
            ValueError, match="Cannot specify both url_only and binary_only"
        ):
            hist.media.videos(url_only=True, binary_only=True)

    def test_media_document_binary_types(self) -> None:
        """Test document detection with various binary media types."""
        pdf_content = BinaryContent(data=b"pdf_data", media_type="application/pdf")
        text_content = BinaryContent(data=b"text_data", media_type="text/plain")

        messages = [
            ModelRequest(parts=[
                UserPromptPart(content=["Documents:", pdf_content, text_content])
            ]),
            ModelResponse(parts=[TextPart(content="Got documents!")]),
        ]

        hist = History(messages)

        documents = hist.media.documents()
        assert len(documents) == 2
        assert pdf_content in documents
        assert text_content in documents

    def test_media_documents_filtering(self) -> None:
        """Test document filtering with url_only and binary_only parameters."""
        doc_url = DocumentUrl(url="https://example.com/document.pdf")
        pdf_content = BinaryContent(data=b"pdf_data", media_type="application/pdf")
        text_content = BinaryContent(data=b"text_data", media_type="text/plain")

        messages = [
            ModelRequest(parts=[
                UserPromptPart(
                    content=["Documents:", doc_url, pdf_content, text_content]
                )
            ]),
            ModelResponse(parts=[TextPart(content="Got documents!")]),
        ]

        hist = History(messages)

        # Test all documents
        all_docs = hist.media.documents()
        assert len(all_docs) == 3
        assert doc_url in all_docs
        assert pdf_content in all_docs
        assert text_content in all_docs

        # Test url_only - should only return DocumentUrl
        url_only_docs = hist.media.documents(url_only=True)
        assert len(url_only_docs) == 1
        assert doc_url in url_only_docs
        assert pdf_content not in url_only_docs
        assert text_content not in url_only_docs

        # Test binary_only - should only return BinaryContent documents
        binary_only_docs = hist.media.documents(binary_only=True)
        assert len(binary_only_docs) == 2
        assert doc_url not in binary_only_docs
        assert pdf_content in binary_only_docs
        assert text_content in binary_only_docs

    def test_media_videos_filtering(self) -> None:
        """Test video filtering with url_only and binary_only parameters."""
        video_url = VideoUrl(url="https://example.com/video.mp4")
        video_content = BinaryContent(data=b"video_data", media_type="video/mp4")

        messages = [
            ModelRequest(parts=[
                UserPromptPart(content=["Videos:", video_url, video_content])
            ]),
            ModelResponse(parts=[TextPart(content="Got videos!")]),
        ]

        hist = History(messages)

        # Test all videos
        all_videos = hist.media.videos()
        assert len(all_videos) == 2
        assert video_url in all_videos
        assert video_content in all_videos

        # Test url_only - should only return VideoUrl
        url_only_videos = hist.media.videos(url_only=True)
        assert len(url_only_videos) == 1
        assert video_url in url_only_videos
        assert video_content not in url_only_videos

        # Test binary_only - should only return BinaryContent videos
        binary_only_videos = hist.media.videos(binary_only=True)
        assert len(binary_only_videos) == 1
        assert video_url not in binary_only_videos
        assert video_content in binary_only_videos

    def test_media_in_repr(self) -> None:
        """Test that media items are included in History repr."""
        image_url = ImageUrl(url="https://example.com/image.jpg")
        messages = [
            ModelRequest(parts=[
                UserPromptPart(content=["Look:", image_url])
            ]),
            ModelResponse(parts=[TextPart(content="Nice!")]),
        ]

        hist = History(messages)
        repr_str = repr(hist)

        assert "1 turn" in repr_str
        assert "1 media item" in repr_str


class TestRealConversations:
    """Test with real conversation data from test_conversations folder."""

    @pytest.fixture
    def conversations_dir(self) -> Path:
        """Get the test conversations directory."""
        return Path(__file__).parent / "test_conversations"

    def load_conversation(self, path: Path) -> list[ModelMessage]:
        """Load a conversation from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return ModelMessagesTypeAdapter.validate_python(data)

    def test_simple_conversation(self, conversations_dir: Path) -> None:
        """Test with a simple single-turn conversation."""
        if not (conversations_dir / "single_turn_simple.json").exists():
            pytest.skip("Test conversation file not found")

        messages = self.load_conversation(conversations_dir / "single_turn_simple.json")
        hist = History(messages)

        # Should have at least one user and one AI message
        assert len(hist.user.all()) >= 1
        assert len(hist.ai.all()) >= 1
        total_tokens = hist.usage().total_tokens
        assert total_tokens is not None and total_tokens > 0

    def test_tool_conversation(self, conversations_dir: Path) -> None:
        """Test with a conversation using tools."""
        if not (conversations_dir / "dice_game_with_tools.json").exists():
            pytest.skip("Test conversation file not found")

        messages = self.load_conversation(
            conversations_dir / "dice_game_with_tools.json"
        )
        hist = History(messages)

        # Should have tool calls and returns
        tool_calls = hist.tools.calls().all()
        tool_returns = hist.tools.returns().all()

        assert len(tool_calls) > 0
        assert len(tool_returns) > 0

        # Check specific tools if we know them
        dice_returns = hist.tools.returns(name="roll_dice").all()
        if dice_returns:
            assert all(r.tool_name == "roll_dice" for r in dice_returns)

    def test_multi_turn_conversation(self, conversations_dir: Path) -> None:
        """Test with a multi-turn conversation."""
        if not (conversations_dir / "simple_multiturn.json").exists():
            pytest.skip("Test conversation file not found")

        messages = self.load_conversation(conversations_dir / "simple_multiturn.json")
        hist = History(messages)

        user_messages = hist.user.all()
        ai_messages = hist.ai.all()

        # Multi-turn should have multiple exchanges
        assert len(user_messages) > 1
        assert len(ai_messages) > 1

        # Last messages should exist
        assert hist.user.last() is not None
        assert hist.ai.last() is not None


class TestTypeAnnotations:
    """Test that type annotations work correctly."""

    def test_type_inference(self) -> None:
        """Test that IDEs can infer types correctly."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi!")]),
        ]

        hist = History(messages)

        # These should all have correct types
        user_part = hist.user.last()
        ai_part = hist.ai.last()
        usage: Usage = hist.usage()

        # Verify types at runtime
        assert user_part is None or isinstance(user_part, UserPromptPart)
        assert ai_part is None or isinstance(ai_part, TextPart)
        assert isinstance(usage, Usage)


class TestConvenienceImports:
    """Test convenience import patterns."""

    def test_ph_alias_import(self) -> None:
        """Test importing as ph alias."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi!")]),
        ]

        # Should work the same as direct import
        hist = ph.History(messages)
        assert len(hist.user.all()) == 1


class TestFileLoading:
    """Test loading history from JSON files."""

    def test_load_from_file_path_string(self) -> None:
        """Test loading history from a file path string."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there!")]),
        ]

        # Create a temporary file with serialized messages
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(to_jsonable_python(messages), f)
            temp_path = f.name

        try:
            # Load from string path
            hist = History(temp_path)
            assert len(hist.all_messages()) == 2
            user_last = hist.user.last()
            ai_last = hist.ai.last()
            assert user_last is not None and user_last.content == "Hello"
            assert ai_last is not None and ai_last.content == "Hi there!"
        finally:
            Path(temp_path).unlink()

    def test_load_from_path_object(self) -> None:
        """Test loading history from a Path object."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="What's the weather?")]),
            ModelResponse(parts=[
                ToolCallPart(
                    tool_name="get_weather",
                    args={"city": "London"},
                    tool_call_id="call_123"
                )
            ]),
            ModelRequest(parts=[
                ToolReturnPart(
                    tool_name="get_weather",
                    content="Rainy, 15°C",
                    tool_call_id="call_123"
                )
            ]),
            ModelResponse(parts=[TextPart(content="It's rainy and 15°C in London.")]),
        ]

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(to_jsonable_python(messages), f)
            temp_path = Path(f.name)

        try:
            # Load from Path object
            hist = History(temp_path)
            assert len(hist.all_messages()) == 4
            user_last = hist.user.last()
            assert user_last is not None and user_last.content == "What's the weather?"
            assert len(hist.tools.calls().all()) == 1
            tool_return = hist.tools.returns().last()
            assert tool_return is not None
            assert hasattr(tool_return, 'content') and tool_return.content == "Rainy, 15°C"
        finally:
            temp_path.unlink()

    def test_load_from_nonexistent_file(self) -> None:
        """Test error handling for non-existent file."""
        with pytest.raises(ValueError, match="does not exist"):
            History("/path/that/does/not/exist.json")

    def test_load_from_nonexistent_path_object(self) -> None:
        """Test error handling for non-existent Path object."""
        with pytest.raises(ValueError, match="does not exist"):
            History(Path("/path/that/does/not/exist.json"))

    def test_load_from_invalid_json_file(self) -> None:
        """Test error handling for invalid JSON content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Failed to load messages"):
                History(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_from_invalid_message_format(self) -> None:
        """Test error handling for valid JSON but invalid message format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"invalid": "message", "format": True}], f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Failed to load messages"):
                History(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_from_path_object_invalid_json(self) -> None:
        """Test error handling for Path object with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Failed to load messages"):
                History(temp_path)
        finally:
            temp_path.unlink()

    def test_roundtrip_serialization(self) -> None:
        """Test that messages can be serialized and deserialized correctly."""
        original_messages = [
            ModelRequest(parts=[UserPromptPart(content="Tell me a joke")]),
            ModelResponse(
                parts=[TextPart(content="Why did the chicken cross the road?")],
                usage=Usage(
                    requests=1,
                    request_tokens=10,
                    response_tokens=15,
                    total_tokens=25
                )
            ),
            ModelRequest(parts=[UserPromptPart(content="I don't know, why?")]),
            ModelResponse(
                parts=[TextPart(content="To get to the other side!")],
                usage=Usage(
                    requests=1,
                    request_tokens=20,
                    response_tokens=10,
                    total_tokens=30
                )
            ),
        ]

        # Serialize to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(to_jsonable_python(original_messages), f)
            temp_path = f.name

        try:
            # Load from file
            hist = History(temp_path)

            # Verify content
            assert len(hist.user.all()) == 2
            assert len(hist.ai.all()) == 2
            assert hist.user.all()[0].content == "Tell me a joke"
            ai_last = hist.ai.last()
            assert (
                ai_last is not None and ai_last.content == "To get to the other side!"
            )

            # Verify usage data survived
            usage = hist.usage()
            assert usage.total_tokens == 55
            assert usage.requests == 2
        finally:
            Path(temp_path).unlink()


class TestRepr:
    """Test the __repr__ method for nice summaries."""

    def test_repr_simple_conversation(self) -> None:
        """Test repr with a simple conversation."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(
                parts=[TextPart(content="Hi!")],
                usage=Usage(total_tokens=50)
            ),
        ]

        hist = History(messages)
        assert repr(hist) == "History(1 turn, 50 tokens)"

    def test_repr_multi_turn(self) -> None:
        """Test repr with multiple turns."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi!")]),
            ModelRequest(parts=[UserPromptPart(content="How are you?")]),
            ModelResponse(parts=[TextPart(content="I'm great!")]),
            ModelRequest(parts=[UserPromptPart(content="Nice!")]),
            ModelResponse(parts=[TextPart(content="Thanks!")]),
        ]

        hist = History(messages)
        # When tokens are 0, they're not included in repr
        assert repr(hist) == "History(3 turns)"

    def test_repr_with_tools(self) -> None:
        """Test repr with tool calls."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Roll dice")]),
            ModelResponse(parts=[
                ToolCallPart(
                    tool_name="roll_dice",
                    args={},
                    tool_call_id="1"
                )
            ]),
            ModelRequest(parts=[
                ToolReturnPart(
                    tool_name="roll_dice",
                    content="6",
                    tool_call_id="1"
                )
            ]),
            ModelResponse(
                parts=[TextPart(content="You rolled a 6!")],
                usage=Usage(total_tokens=100)
            ),
        ]

        hist = History(messages)
        assert repr(hist) == "History(1 turn, 100 tokens, 1 tool call)"

    def test_repr_multiple_tool_calls(self) -> None:
        """Test repr with multiple tool calls."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Check weather and time")]),
            ModelResponse(parts=[
                ToolCallPart(tool_name="get_weather", args={}, tool_call_id="1"),
                ToolCallPart(tool_name="get_time", args={}, tool_call_id="2"),
            ]),
            ModelRequest(parts=[
                ToolReturnPart(
                    tool_name="get_weather", content="Sunny", tool_call_id="1"
                ),
                ToolReturnPart(tool_name="get_time", content="3pm", tool_call_id="2"),
            ]),
            ModelResponse(parts=[TextPart(content="It's sunny and 3pm")]),
        ]

        hist = History(messages)
        # When tokens are 0, they're not included in repr
        assert repr(hist) == "History(1 turn, 2 tool calls)"

    def test_repr_empty_history(self) -> None:
        """Test repr with empty history."""
        hist = History([])
        assert repr(hist) == "History(0 turns)"

    def test_repr_no_user_messages(self) -> None:
        """Test repr with only system/AI messages."""
        messages = [
            ModelRequest(parts=[SystemPromptPart(content="Be helpful")]),
            ModelResponse(parts=[TextPart(content="I'll be helpful!")]),
        ]

        hist = History(messages)
        # No user messages means 0 turns
        assert repr(hist) == "History(0 turns)"

    def test_str_same_as_repr(self) -> None:
        """Test that str() gives same result as repr()."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(
                parts=[TextPart(content="Hi!")],
                usage=Usage(total_tokens=42)
            ),
        ]

        hist = History(messages)
        assert str(hist) == repr(hist)
        assert str(hist) == "History(1 turn, 42 tokens)"
