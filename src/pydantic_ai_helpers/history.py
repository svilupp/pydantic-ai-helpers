"""Fluent history accessor for PydanticAI conversations.

This module provides a simple, chainable API for accessing message history
from PydanticAI agents. Instead of manually iterating through messages and
checking types, you get autocomplete-friendly accessors:

    hist = History(result)
    last_user_msg = hist.user.last()
    all_ai_responses = hist.ai.all()
    dice_roll = hist.tools.returns(name="roll_dice").last()

It's boring, it's obvious, and you'll wonder how you lived without it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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

if TYPE_CHECKING:
    from pydantic_ai._result import (  # type: ignore[import-not-found]
        RunResult,
        StreamedRunResult,
    )


def _iter_messages(source: Any) -> list[ModelMessage]:
    """Extract messages from various PydanticAI result types.

    Parameters
    ----------
    source : Any
        A RunResult, StreamedRunResult, iterable of ModelMessage,
        or a path to a JSON file.

    Returns
    -------
    list[ModelMessage]
        Concrete list of messages.

    Raises
    ------
    TypeError
        If source is not a supported type.
    ValueError
        If file path doesn't exist or contains invalid data.
    """
    # Check if it's a file path string
    if isinstance(source, str):
        path = Path(source)
        if path.exists() and path.is_file():
            try:
                with open(path) as f:
                    data = json.load(f)
                return ModelMessagesTypeAdapter.validate_python(data)
            except Exception as e:
                raise ValueError(
                    f"Failed to load messages from file '{path}': {e}"
                ) from e
        else:
            raise ValueError(f"File path '{source}' does not exist or is not a file")

    # Check if it's a Path object
    if isinstance(source, Path):
        if source.exists() and source.is_file():
            try:
                with open(source) as f:
                    data = json.load(f)
                return ModelMessagesTypeAdapter.validate_python(data)
            except Exception as e:
                raise ValueError(
                    f"Failed to load messages from file '{source}': {e}"
                ) from e
        else:
            raise ValueError(f"File path '{source}' does not exist or is not a file")

    # Existing logic for RunResult/StreamedRunResult
    if hasattr(source, "all_messages"):
        return list(source.all_messages())

    # Existing logic for iterables
    if isinstance(source, Iterable):
        messages = list(source)
        if messages and all(
            isinstance(m, ModelRequest | ModelResponse) for m in messages
        ):
            return messages
        if not messages:
            return []

    raise TypeError(
        "History() expects a RunResult, StreamedRunResult, an iterable of "
        f"ModelMessage, or a file path. Got {type(source).__name__} instead."
    )


def _parts_for_role(
    messages: list[ModelMessage], role: str
) -> Generator[UserPromptPart | TextPart | SystemPromptPart, None, None]:
    """Yield message parts that match a logical chat role.

    Parameters
    ----------
    messages : list[ModelMessage]
        Messages to filter.
    role : str
        One of "user", "ai", or "system".

    Yields
    ------
    Union[UserPromptPart, TextPart, SystemPromptPart]
        Parts matching the specified role.
    """
    for msg in messages:
        for part in msg.parts:
            if (
                (role == "user" and isinstance(part, UserPromptPart))
                or (role == "ai" and isinstance(part, TextPart))
                or (role == "system" and isinstance(part, SystemPromptPart))
            ):
                yield part


def _tool_parts(
    messages: list[ModelMessage], kind: str
) -> Generator[ToolCallPart | ToolReturnPart, None, None]:
    """Yield tool-related parts from messages.

    Parameters
    ----------
    messages : list[ModelMessage]
        Messages to search.
    kind : str
        Either "call" or "return".

    Yields
    ------
    Union[ToolCallPart, ToolReturnPart]
        Tool parts of the specified kind.
    """
    cls = ToolCallPart if kind == "call" else ToolReturnPart
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, cls):
                yield cast(ToolCallPart | ToolReturnPart, part)


class RoleView:
    """Fluent accessor for parts from a specific chat role.

    Provides filtered access to message parts based on their role
    (user, ai, or system). Supports both bulk access and convenient
    access to the last message.

    Parameters
    ----------
    messages : list[ModelMessage]
        Messages to filter.
    role : str
        Role to filter for ("user", "ai", or "system").

    Examples
    --------
    >>> hist = History(result)
    >>> hist.user.last()  # Get last user message
    UserPromptPart(content="Hello!")
    >>> len(hist.ai.all())  # Count AI responses
    3
    """

    def __init__(self, messages: list[ModelMessage], role: str):
        self._parts = list(_parts_for_role(messages, role))

    def all(self) -> list[UserPromptPart | TextPart | SystemPromptPart]:
        """Get all parts for this role.

        Returns
        -------
        list[Union[UserPromptPart, TextPart, SystemPromptPart]]
            All message parts matching the role.
        """
        return list(self._parts)

    def last(self) -> UserPromptPart | TextPart | SystemPromptPart | None:
        """Get the most recent part for this role.

        Returns
        -------
        Union[UserPromptPart, TextPart, SystemPromptPart, None]
            The last part, or None if no parts exist.
        """
        return self._parts[-1] if self._parts else None


class ToolPartView:
    """Accessor for tool calls or returns, optionally filtered by name.

    Provides access to tool-related message parts with optional
    filtering by tool name.

    Parameters
    ----------
    parts : list[Union[ToolCallPart, ToolReturnPart]]
        Tool parts to wrap.
    name : Union[str, None]
        If provided, only include parts for this tool name.

    Examples
    --------
    >>> hist = History(result)
    >>> hist.tools.calls(name="calculator").last()
    ToolCallPart(tool_name="calculator", ...)
    """

    def __init__(self, parts: list[ToolCallPart | ToolReturnPart], name: str | None):
        if name is not None:
            parts = [p for p in parts if p.tool_name == name]
        self._parts = parts

    def all(self) -> list[ToolCallPart | ToolReturnPart]:
        """Get all tool parts (optionally filtered by name).

        Returns
        -------
        list[Union[ToolCallPart, ToolReturnPart]]
            All matching tool parts.
        """
        return list(self._parts)

    def last(self) -> ToolCallPart | ToolReturnPart | None:
        """Get the most recent tool part.

        Returns
        -------
        Union[ToolCallPart, ToolReturnPart, None]
            The last tool part, or None if none exist.
        """
        return self._parts[-1] if self._parts else None


class ToolsView:
    """Root accessor for tool-related message parts.

    Provides methods to access tool calls and returns, with
    optional filtering by tool name.

    Parameters
    ----------
    messages : list[ModelMessage]
        Messages to search for tool parts.

    Examples
    --------
    >>> hist = History(result)
    >>> hist.tools.calls().all()  # All tool calls
    [ToolCallPart(...), ToolCallPart(...)]
    >>> hist.tools.returns(name="weather").last()  # Last weather return
    ToolReturnPart(tool_name="weather", ...)
    """

    def __init__(self, messages: list[ModelMessage]):
        self._messages = messages

    def calls(self, *, name: str | None = None) -> ToolPartView:
        """Access tool call parts.

        Parameters
        ----------
        name : Union[str, None]
            If provided, only include calls to this tool.

        Returns
        -------
        ToolPartView
            View of tool call parts.
        """
        return ToolPartView(list(_tool_parts(self._messages, "call")), name)

    def returns(self, *, name: str | None = None) -> ToolPartView:
        """Access tool return parts.

        Parameters
        ----------
        name : Union[str, None]
            If provided, only include returns from this tool.

        Returns
        -------
        ToolPartView
            View of tool return parts.
        """
        return ToolPartView(list(_tool_parts(self._messages, "return")), name)


# Type alias for all media content types
MediaContent = ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent


def _extract_media_from_messages(messages: list[ModelMessage]) -> list[MediaContent]:
    """Extract all media content from messages.

    Parameters
    ----------
    messages : list[ModelMessage]
        Messages to search for media content.

    Returns
    -------
    list[MediaContent]
        All media content found in the messages.
    """
    media_items = []

    for msg in messages:
        for part in msg.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, list):
                for item in part.content:
                    if isinstance(
                        item,
                        ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent,
                    ):
                        media_items.append(item)

    return media_items


class MediaView:
    """Accessor for media content in conversation history.

    Provides filtered access to media content from UserPromptPart messages,
    including both URL-based media (ImageUrl, AudioUrl, etc.) and binary
    content (BinaryContent).

    Parameters
    ----------
    messages : list[ModelMessage]
        Messages to search for media content.

    Examples
    --------
    >>> hist = History(result)
    >>> hist.media.all()  # All media content
    [ImageUrl(...), BinaryContent(...)]
    >>> hist.media.images()  # Only images
    [ImageUrl(...), BinaryContent(media_type='image/jpeg')]
    >>> hist.media.images(url_only=True)  # Only ImageUrl objects
    [ImageUrl(...)]
    """

    def __init__(self, messages: list[ModelMessage]):
        self._media_items = _extract_media_from_messages(messages)

    def all(self) -> list[MediaContent]:
        """Get all media content from the conversation.

        Returns
        -------
        list[MediaContent]
            All media content found in user messages.
        """
        return list(self._media_items)

    def last(self) -> MediaContent | None:
        """Get the most recent media content.

        Returns
        -------
        Union[MediaContent, None]
            The last media content, or None if no media exists.
        """
        return self._media_items[-1] if self._media_items else None

    def images(
        self, *, url_only: bool = False, binary_only: bool = False
    ) -> list[ImageUrl | BinaryContent]:
        """Get all image content.

        Parameters
        ----------
        url_only : bool, default False
            If True, only return ImageUrl objects.
        binary_only : bool, default False
            If True, only return BinaryContent objects with image media types.

        Returns
        -------
        list[Union[ImageUrl, BinaryContent]]
            All image content matching the filter criteria.
        """
        if url_only and binary_only:
            raise ValueError("Cannot specify both url_only and binary_only")

        images: list[ImageUrl | BinaryContent] = []
        for item in self._media_items:
            if isinstance(item, ImageUrl):
                if not binary_only:
                    images.append(item)
            elif (
                isinstance(item, BinaryContent)
                and item.media_type.startswith("image/")
                and not url_only
            ):
                images.append(item)

        return images

    def audio(
        self, *, url_only: bool = False, binary_only: bool = False
    ) -> list[AudioUrl | BinaryContent]:
        """Get all audio content.

        Parameters
        ----------
        url_only : bool, default False
            If True, only return AudioUrl objects.
        binary_only : bool, default False
            If True, only return BinaryContent objects with audio media types.

        Returns
        -------
        list[Union[AudioUrl, BinaryContent]]
            All audio content matching the filter criteria.
        """
        if url_only and binary_only:
            raise ValueError("Cannot specify both url_only and binary_only")

        audio: list[AudioUrl | BinaryContent] = []
        for item in self._media_items:
            if isinstance(item, AudioUrl):
                if not binary_only:
                    audio.append(item)
            elif (
                isinstance(item, BinaryContent)
                and item.media_type.startswith("audio/")
                and not url_only
            ):
                audio.append(item)

        return audio

    def documents(
        self, *, url_only: bool = False, binary_only: bool = False
    ) -> list[DocumentUrl | BinaryContent]:
        """Get all document content.

        Parameters
        ----------
        url_only : bool, default False
            If True, only return DocumentUrl objects.
        binary_only : bool, default False
            If True, only return BinaryContent objects with document media types.

        Returns
        -------
        list[Union[DocumentUrl, BinaryContent]]
            All document content matching the filter criteria.
        """
        if url_only and binary_only:
            raise ValueError("Cannot specify both url_only and binary_only")

        documents: list[DocumentUrl | BinaryContent] = []
        for item in self._media_items:
            if isinstance(item, DocumentUrl):
                if not binary_only:
                    documents.append(item)
            elif (
                isinstance(item, BinaryContent)
                and (
                    item.media_type.startswith("application/")
                    or item.media_type.startswith("text/")
                )
                and not url_only
            ):
                documents.append(item)

        return documents

    def videos(
        self, *, url_only: bool = False, binary_only: bool = False
    ) -> list[VideoUrl | BinaryContent]:
        """Get all video content.

        Parameters
        ----------
        url_only : bool, default False
            If True, only return VideoUrl objects.
        binary_only : bool, default False
            If True, only return BinaryContent objects with video media types.

        Returns
        -------
        list[Union[VideoUrl, BinaryContent]]
            All video content matching the filter criteria.
        """
        if url_only and binary_only:
            raise ValueError("Cannot specify both url_only and binary_only")

        videos: list[VideoUrl | BinaryContent] = []
        for item in self._media_items:
            if isinstance(item, VideoUrl):
                if not binary_only:
                    videos.append(item)
            elif (
                isinstance(item, BinaryContent)
                and item.media_type.startswith("video/")
                and not url_only
            ):
                videos.append(item)

        return videos

    def by_type(self, media_type: type) -> list[MediaContent]:
        """Get all media content of a specific type.

        Parameters
        ----------
        media_type : type
            The type to filter for (e.g., ImageUrl, BinaryContent).

        Returns
        -------
        list[MediaContent]
            All media content of the specified type.
        """
        return [item for item in self._media_items if isinstance(item, media_type)]


class History:
    """Fluent wrapper around PydanticAI message history.

    Wraps any object that exposes all_messages() (like RunResult or
    StreamedRunResult), a plain list of ModelMessage objects, or loads
    from a JSON file. Provides chainable, autocomplete-friendly access
    to messages.

    Parameters
    ----------
    result_or_messages : Union[
        RunResult, StreamedRunResult, Iterable[ModelMessage], str, Path
    ]
        The source of messages to wrap. Can be a RunResult, StreamedRunResult,
        list of messages, or a path to a JSON file containing serialized messages.

    Attributes
    ----------
    user : RoleView
        Access to user messages.
    ai : RoleView
        Access to AI/assistant messages.
    system : RoleView
        Access to system messages.
    tools : ToolsView
        Access to tool calls and returns.
    media : MediaView
        Access to media content in user messages.

    Examples
    --------
    >>> from pydantic_ai import Agent
    >>> agent = Agent("openai:gpt-4o")
    >>> result = agent.run_sync("Tell me a joke")
    >>> hist = History(result)
    >>>
    >>> # Access the last user message
    >>> hist.user.last().content
    "Tell me a joke"
    >>>
    >>> # Get all AI responses
    >>> for response in hist.ai.all():
    ...     print(response.content)
    >>>
    >>> # Check token usage
    >>> hist.usage().total_tokens
    127
    >>>
    >>> # Load from file
    >>> hist = History("conversation.json")
    >>> print(hist)  # Shows nice summary
    History(2 turns, 127 tokens, 1 tool call)

    Notes
    -----
    The History wrapper is immutable and does not modify the
    original messages. All view objects are created lazily to
    maintain performance.
    """

    def __init__(
        self,
        result_or_messages: RunResult
        | StreamedRunResult
        | Iterable[ModelMessage]
        | str
        | Path,
    ):
        self._messages = _iter_messages(result_or_messages)

        # Create views lazily for better performance
        self.user: RoleView = RoleView(self._messages, "user")
        self.ai: RoleView = RoleView(self._messages, "ai")
        self.system: RoleView = RoleView(self._messages, "system")
        self.tools: ToolsView = ToolsView(self._messages)
        self.media: MediaView = MediaView(self._messages)

    def all_messages(self) -> list[ModelMessage]:
        """Get the raw list of messages.

        Returns
        -------
        list[ModelMessage]
            All messages in their original form.

        Notes
        -----
        This returns a copy of the message list to prevent
        accidental modification.
        """
        return list(self._messages)

    def usage(self) -> Usage:
        """Aggregate token usage across all model responses.

        Walks through all ModelResponse messages in the history and
        sums up their Usage data. All numeric fields are added
        together, and details dictionaries are merged key-wise.

        Returns
        -------
        Usage
            Combined usage statistics. Fields will be 0 if no
            usage data exists.

        Examples
        --------
        >>> hist = History(agent.run_sync("Hello"))
        >>> tokens = hist.usage()
        >>> print(f"Total tokens: {tokens.total_tokens}")
        Total tokens: 127
        >>> print(f"Cost estimate: ${tokens.total_tokens * 0.00002:.4f}")
        Cost estimate: $0.0025

        See Also
        --------
        tokens : Alias for this method.
        """
        totals: dict[str, int] = defaultdict(int)
        details: dict[str, int] = defaultdict(int)

        for msg in self._messages:
            if isinstance(msg, ModelResponse) and msg.usage:
                u: Usage = msg.usage
                totals["requests"] += u.requests or 0
                totals["request_tokens"] += u.request_tokens or 0
                totals["response_tokens"] += u.response_tokens or 0
                totals["total_tokens"] += u.total_tokens or 0
                if u.details:
                    for k, v in u.details.items():
                        details[k] += v

        return Usage(**totals, details=dict(details) or None)

    def tokens(self) -> Usage:
        """Alias for usage() - many developers think "tokens" first.

        Returns
        -------
        Usage
            Combined usage statistics.

        See Also
        --------
        usage : The primary method this aliases.
        """
        return self.usage()

    def system_prompt(self) -> SystemPromptPart | None:
        """Get the first system prompt from the conversation.

        Searches through all messages to find the first SystemPromptPart,
        which is typically included in the first message if present.

        Returns
        -------
        Union[SystemPromptPart, None]
            The first system prompt found, or None if no system prompt exists.

        Examples
        --------
        >>> hist = History(result)
        >>> prompt = hist.system_prompt()
        >>> if prompt:
        ...     print(prompt.content)
        "You are a helpful AI assistant."
        """
        for msg in self._messages:
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    return part
        return None

    def __repr__(self) -> str:
        """Return a nice summary of the conversation.

        Returns
        -------
        str
            A summary showing turns, tokens, and tool usage.

        Examples
        --------
        >>> hist = History(result)
        >>> print(hist)
        History(2 turns, 127 tokens, 3 tool calls)
        """
        # Count turns (user messages typically indicate turns)
        turns = len(self.user.all())

        # Get token usage
        usage = self.usage()
        total_tokens = usage.total_tokens or 0

        # Count tool interactions
        tool_calls = len(self.tools.calls().all())

        # Count media items
        media_items = len(self.media.all())

        # Build summary parts
        parts = [f"{turns} turn{'s' if turns != 1 else ''}"]

        if total_tokens > 0:
            parts.append(f"{total_tokens} tokens")

        if tool_calls > 0:
            parts.append(f"{tool_calls} tool call{'s' if tool_calls != 1 else ''}")

        if media_items > 0:
            parts.append(f"{media_items} media item{'s' if media_items != 1 else ''}")

        # Create the summary string
        return f"History({', '.join(parts)})"
