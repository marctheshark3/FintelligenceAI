#!/usr/bin/env python3
"""
FintelligenceAI Local Mode Demo

This script demonstrates how to use FintelligenceAI in local-only mode
with Ollama for completely offline operation.

Features demonstrated:
- Local configuration setup
- Ollama model usage with DSPy
- Local embeddings
- RAG pipeline with local components
- CLI integration examples
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from fintelligence_ai.config import get_settings
from fintelligence_ai.core.ollama import (
    OllamaClient,
    get_ollama_dspy_model,
    get_ollama_embedding_service,
)


def print_banner():
    """Print demo banner."""
    print("=" * 60)
    print("🦙 FintelligenceAI Local Mode Demo")
    print("=" * 60)
    print()


def print_section(title: str):
    """Print section header."""
    print(f"\n📋 {title}")
    print("-" * (len(title) + 4))


async def demo_configuration():
    """Demonstrate configuration setup for local mode."""
    print_section("Configuration Setup")

    # Show current configuration
    settings = get_settings()
    print(f"Current Mode: {'Local' if settings.dspy.local_mode else 'Cloud'}")
    print(f"Model Provider: {settings.dspy.model_provider}")
    print(f"Ollama Host: {settings.ollama.host}:{settings.ollama.port}")
    print(f"Ollama Model: {settings.ollama.model}")
    print(f"Embedding Model: {settings.ollama.embedding_model}")

    # Show how to enable local mode
    print("\n💡 To enable local mode, set these environment variables:")
    print("   export DSPY_LOCAL_MODE=true")
    print("   export DSPY_MODEL_PROVIDER=ollama")
    print("   export OLLAMA_MODEL=llama3.2")
    print("   export OLLAMA_EMBEDDING_MODEL=nomic-embed-text")


async def demo_ollama_client():
    """Demonstrate Ollama client functionality."""
    print_section("Ollama Client Demo")

    client = OllamaClient()

    # Check if Ollama is available
    print("🔍 Checking Ollama server availability...")
    is_available = client.is_available()
    print(f"   Status: {'✅ Available' if is_available else '❌ Not available'}")

    if not is_available:
        print("   💡 Make sure Ollama is installed and running:")
        print("      curl -fsSL https://ollama.ai/install.sh | sh")
        print("      ollama serve")
        return False

    # List available models
    print("\n📋 Available models:")
    models = client.list_models()
    if models:
        for model in models:
            name = model.get("name", "unknown")
            size = model.get("size", 0)
            size_gb = size / (1024**3) if size else 0
            print(f"   • {name} ({size_gb:.1f}GB)")
    else:
        print("   No models found. Try: ollama pull llama3.2")

    return True


async def demo_text_generation():
    """Demonstrate text generation with local models."""
    print_section("Text Generation Demo")

    try:
        # Create DSPy model instance
        model = get_ollama_dspy_model(
            model="llama3:latest",
            temperature=0.1,  # Use a model that exists
        )

        print("🤖 Generating text with local model...")
        prompt = "Explain what a blockchain is in simple terms."
        print(f"   Prompt: {prompt}")

        response = model.basic_request(prompt)
        print(f"   Response: {response[:200]}...")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        print("   💡 Make sure you have pulled the required model:")
        print("      ollama pull llama3")


async def demo_embeddings():
    """Demonstrate local embeddings."""
    print_section("Local Embeddings Demo")

    try:
        # Create embedding service
        embedding_service = get_ollama_embedding_service(model="nomic-embed-text")

        print("🔗 Generating embeddings with local model...")
        texts = [
            "Ergo is a proof-of-work cryptocurrency",
            "Smart contracts enable programmable money",
            "Blockchain technology ensures decentralization",
        ]

        for text in texts:
            print(f"   Text: {text}")
            embedding = embedding_service.embed_text(text)
            print(f"   Embedding dimensions: {len(embedding)}")
            print(f"   Sample values: {embedding[:5]}...")

    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        print("   💡 Make sure you have pulled the embedding model:")
        print("      ollama pull nomic-embed-text")


async def demo_dspy_integration():
    """Demonstrate DSPy integration with local models."""
    print_section("DSPy Integration Demo")

    try:
        import dspy

        # Configure DSPy with Ollama
        model = get_ollama_dspy_model(
            model="llama3:latest",
            temperature=0.1,  # Use a model that exists
        )

        # Test the model directly first
        print("🧠 Testing model directly...")
        question = "What makes Ergo blockchain unique?"
        print(f"   Question: {question}")

        try:
            # Direct model call
            answer = model.generate(f"Question: {question}\nAnswer:")
            print(f"   Answer: {answer[:200]}...")

            print("\n🧠 Testing with DSPy ChainOfThought...")
            # Now try with DSPy
            dspy.configure(lm=model)

            # Simple DSPy signature
            class SimpleQA(dspy.Signature):
                """Answer questions about blockchain technology."""

                question = dspy.InputField()
                answer = dspy.OutputField(desc="A clear, concise answer")

            # Create and use a DSPy module
            qa_module = dspy.ChainOfThought(SimpleQA)
            result = qa_module(question=question)
            print(f"   DSPy Answer: {result.answer[:200]}...")

        except Exception as inner_e:
            print(f"   ⚠️  Direct call failed: {inner_e}")
            # Try fallback method
            try:
                answer = model(question)
                print(f"   Fallback Answer: {answer[:200]}...")
            except Exception as fallback_e:
                print(f"   ❌ Fallback also failed: {fallback_e}")

    except Exception as e:
        print(f"❌ DSPy integration failed: {e}")
        print("   💡 This requires a working Ollama setup with models")


def demo_cli_commands():
    """Demonstrate CLI commands for local mode."""
    print_section("CLI Commands Demo")

    print("🖥️  Local mode CLI commands:")
    print("   # Check system health")
    print("   python -m fintelligence_ai.cli health")
    print()
    print("   # Check local mode status")
    print("   python -m fintelligence_ai.cli local status")
    print()
    print("   # Setup local mode with recommended models")
    print("   python -m fintelligence_ai.cli local setup")
    print()
    print("   # Pull a specific model")
    print("   python -m fintelligence_ai.cli local pull llama3.2")
    print()
    print("   # Start server in local mode")
    print("   DSPY_LOCAL_MODE=true python -m fintelligence_ai.cli serve")


def demo_docker_setup():
    """Show Docker setup for local mode."""
    print_section("Docker Setup Demo")

    print("🐳 Docker setup for local mode:")
    print("   # Use the local-only Docker Compose file")
    print("   docker-compose -f docker-compose.local.yml up -d")
    print()
    print("   # This includes:")
    print("   • Ollama server with GPU support")
    print("   • ChromaDB for vector storage")
    print("   • Redis for caching")
    print("   • FintelligenceAI with local mode enabled")
    print()
    print("   # Check status")
    print("   docker-compose -f docker-compose.local.yml ps")


async def demo_performance_tips():
    """Show performance optimization tips."""
    print_section("Performance Optimization Tips")

    print("⚡ Performance optimization for local mode:")
    print()
    print("1. 🖥️  Hardware Recommendations:")
    print("   • GPU: NVIDIA GPU with 8GB+ VRAM (recommended)")
    print("   • RAM: 16GB+ system RAM")
    print("   • Storage: SSD for model storage")
    print()
    print("2. 🔧 Model Selection:")
    print("   • Development: llama3.2 (4GB) or qwen2.5:7b")
    print("   • Production: llama3.1:8b or mistral-nemo")
    print("   • Embeddings: nomic-embed-text or mxbai-embed-large")
    print()
    print("3. ⚙️  Configuration Tuning:")
    print("   • Set OLLAMA_NUM_PARALLEL=2 for concurrent requests")
    print("   • Use keep_alive='30m' for frequently used models")
    print(
        "   • Adjust temperature based on use case (0.1 for factual, 0.7 for creative)"
    )
    print()
    print("4. 📊 Monitoring:")
    print("   • Monitor GPU memory usage: nvidia-smi")
    print("   • Check Ollama logs: ollama logs")
    print("   • Use FintelligenceAI health checks")


async def main():
    """Run the complete demo."""
    print_banner()

    print("This demo shows FintelligenceAI's local-only mode capabilities.")
    print("Local mode eliminates external API dependencies using Ollama.")
    print()

    # Run demo sections
    await demo_configuration()

    ollama_available = await demo_ollama_client()

    if ollama_available:
        await demo_text_generation()
        await demo_embeddings()
        await demo_dspy_integration()

    demo_cli_commands()
    demo_docker_setup()
    await demo_performance_tips()

    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
    print("2. Pull models: ollama pull llama3.2 && ollama pull nomic-embed-text")
    print("3. Enable local mode: export DSPY_LOCAL_MODE=true")
    print("4. Run FintelligenceAI: python -m fintelligence_ai.cli serve")
    print()
    print("📖 See docs/LOCAL_MODE_SETUP.md for detailed setup instructions.")


if __name__ == "__main__":
    # Check if running in local mode
    if os.getenv("DSPY_LOCAL_MODE", "false").lower() == "true":
        print("🦙 Running in local mode!")
    else:
        print("ℹ️  Running in cloud mode. Set DSPY_LOCAL_MODE=true for local mode.")

    asyncio.run(main())
