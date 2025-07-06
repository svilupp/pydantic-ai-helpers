#!/usr/bin/env python3
"""
Example showcasing media content and system prompt features.

This example demonstrates how to:
1. Work with conversations containing media content
2. Extract and analyze system prompts
3. Use the new MediaView and system_prompt() features
"""

from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from pydantic_ai_helpers import History


def create_sample_conversation():
    """Create a sample conversation with media content and system prompt."""
    # Create a conversation with system prompt and media
    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart(
                    content="You are a helpful AI assistant specialized in "
                    "analyzing media content."
                ),
                UserPromptPart(
                    content=[
                        "Please analyze this image and audio file:",
                        ImageUrl(url="https://example.com/vacation-photo.jpg"),
                        AudioUrl(url="https://example.com/voice-memo.wav"),
                    ]
                ),
            ]
        ),
        ModelResponse(
            parts=[
                TextPart(
                    content="I can see a beautiful vacation photo and hear a "
                    "voice memo. The image shows..."
                )
            ]
        ),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "Here's another image with some binary data:",
                        BinaryContent(
                            data=b"fake_image_data_here", media_type="image/png"
                        ),
                    ]
                )
            ]
        ),
        ModelResponse(
            parts=[
                TextPart(
                    content="I've received the binary image data. This appears "
                    "to be a PNG image..."
                )
            ]
        ),
    ]

    return messages


def analyze_conversation(messages):  # noqa: PLR0912, PLR0915
    """Analyze the conversation using History wrapper."""
    print("=== Conversation Analysis with Media and System Prompts ===\n")

    # Create History wrapper
    hist = History(messages)

    # Basic conversation info
    print(f"📊 {hist}")
    print(f"   User messages: {len(hist.user.all())}")
    print(f"   AI responses: {len(hist.ai.all())}")

    # System prompt analysis
    print("\n=== System Prompt Analysis ===")
    system_prompt = hist.system_prompt()
    if system_prompt:
        print(f"✓ System prompt found: {system_prompt.content}")

        # Analyze system prompt content
        content = system_prompt.content.lower()
        if "helpful" in content:
            print("  → Agent configured to be helpful")
        if "media" in content or "image" in content or "audio" in content:
            print("  → Agent specialized for media content")
    else:
        print("✗ No system prompt found")

    # Media content analysis
    print("\n=== Media Content Analysis ===")
    all_media = hist.media.all()
    if all_media:
        print(f"✓ Found {len(all_media)} media items:")

        for i, item in enumerate(all_media, 1):
            print(f"  {i}. {type(item).__name__}")
            if hasattr(item, "url"):
                print(f"     URL: {item.url}")
            elif hasattr(item, "media_type"):
                print(f"     Type: {item.media_type}")
                print(f"     Size: {len(item.data)} bytes")

        # Categorize media
        print(f"\n📷 Images: {len(hist.media.images())}")
        print(f"🎵 Audio: {len(hist.media.audio())}")
        print(f"📄 Documents: {len(hist.media.documents())}")
        print(f"🎬 Videos: {len(hist.media.videos())}")

        # Show latest media
        latest = hist.media.last()
        print(f"\n🔄 Latest media: {type(latest).__name__}")

        # Filter by storage type
        url_media = []
        binary_media = []

        for item in all_media:
            if hasattr(item, "url"):
                url_media.append(item)
            else:
                binary_media.append(item)

        print(f"\n🔗 URL-based media: {len(url_media)}")
        print(f"💾 Binary media: {len(binary_media)}")

    else:
        print("✗ No media content found")

    # Show specific filtering examples
    print("\n=== Advanced Filtering Examples ===")

    # Images only
    images = hist.media.images()
    if images:
        print(f"🖼️  All images ({len(images)}):")
        for img in images:
            if hasattr(img, "url"):
                print(f"   - URL: {img.url}")
            else:
                print(f"   - Binary: {img.media_type}")

    # URL-only images
    url_images = hist.media.images(url_only=True)
    if url_images:
        print(f"🌐 URL-only images: {len(url_images)}")

    # Binary-only images
    binary_images = hist.media.images(binary_only=True)
    if binary_images:
        print(f"💾 Binary-only images: {len(binary_images)}")

    # By type filtering
    from pydantic_ai.messages import BinaryContent, ImageUrl

    image_urls = hist.media.by_type(ImageUrl)
    binary_content = hist.media.by_type(BinaryContent)

    print("\n🔍 By type:")
    print(f"   ImageUrl objects: {len(image_urls)}")
    print(f"   BinaryContent objects: {len(binary_content)}")


def demonstrate_real_world_usage():
    """Show how you might use these features in a real application."""
    print("\n" + "=" * 60)
    print("=== Real-World Usage Examples ===")
    print("=" * 60)

    messages = create_sample_conversation()
    hist = History(messages)

    # Example 1: Agent configuration validation
    print("\n1. 🔧 Agent Configuration Validation")
    system_prompt = hist.system_prompt()
    if system_prompt and "media" in system_prompt.content.lower():
        print("   ✓ Agent properly configured for media analysis")
    else:
        print("   ⚠️  Agent may not be optimized for media content")

    # Example 2: Content moderation
    print("\n2. 🛡️  Content Moderation Check")
    media_count = len(hist.media.all())
    if media_count > 0:
        print(f"   📊 Found {media_count} media items to review")

        # Check for different types
        image_count = len(hist.media.images())
        audio_count = len(hist.media.audio())

        if image_count > 0:
            print(f"   🖼️  {image_count} images need visual review")
        if audio_count > 0:
            print(f"   🎵 {audio_count} audio files need audio review")

    # Example 3: Cost estimation
    print("\n3. 💰 Processing Cost Estimation")
    total_tokens = hist.usage().total_tokens or 0
    media_items = len(hist.media.all())

    # Hypothetical costs
    text_cost = total_tokens * 0.00002  # $0.00002 per token
    media_cost = media_items * 0.01  # $0.01 per media item
    total_cost = text_cost + media_cost

    print(f"   💬 Text processing: ${text_cost:.4f} ({total_tokens} tokens)")
    print(f"   🎭 Media processing: ${media_cost:.4f} ({media_items} items)")
    print(f"   💵 Total estimated cost: ${total_cost:.4f}")

    # Example 4: Analytics for dashboards
    print("\n4. 📈 Analytics Summary")
    conversation_stats = {
        "turns": len(hist.user.all()),
        "tokens": total_tokens,
        "media_items": media_items,
        "has_system_prompt": hist.system_prompt() is not None,
        "media_breakdown": {
            "images": len(hist.media.images()),
            "audio": len(hist.media.audio()),
            "documents": len(hist.media.documents()),
            "videos": len(hist.media.videos()),
        },
    }

    print("   📊 Conversation Analytics:")
    print(f"      • Turns: {conversation_stats['turns']}")
    print(f"      • Tokens: {conversation_stats['tokens']}")
    print(f"      • Media items: {conversation_stats['media_items']}")
    print(
        f"      • System prompt: "
        f"{'Yes' if conversation_stats['has_system_prompt'] else 'No'}"
    )
    print(f"      • Media breakdown: {conversation_stats['media_breakdown']}")


if __name__ == "__main__":
    # Create and analyze a sample conversation
    messages = create_sample_conversation()
    analyze_conversation(messages)

    # Show real-world usage examples
    demonstrate_real_world_usage()

    print("\n" + "=" * 60)
    print("✨ Media and system prompt features demonstrated!")
    print("✨ Ready to analyze your own conversations!")
    print("=" * 60)
