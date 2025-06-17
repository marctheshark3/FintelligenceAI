#!/usr/bin/env python3
"""
Setup script for Ollama embeddings in FintelligenceAI

This script helps users set up Ollama with the correct embedding model
for local knowledge base operations.
"""

import os
import subprocess
import sys
from pathlib import Path


def load_local_env():
    """Load environment configuration for Ollama support."""
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        try:
            import dotenv

            dotenv.load_dotenv(env_path)
            print(f"✅ Loaded configuration from {env_path}")
        except ImportError:
            # If python-dotenv is not available, manually parse the file
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key] = value
            print(f"✅ Loaded configuration manually from {env_path}")
    else:
        print(f"⚠️  No .env file found at {env_path}")
        print("   Please copy env.template to .env and configure your settings")
        return False
    return True


def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            print(f"   Version: {result.stdout.strip()}")
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def check_ollama_running():
    """Check if Ollama service is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("✅ Ollama service is running")
            return True
        else:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_embedding_model(model_name="nomic-embed-text"):
    """Install the embedding model in Ollama."""
    print(f"📦 Installing embedding model: {model_name}")
    print("   This may take a few minutes...")

    try:
        # Pull the model
        result = subprocess.run(
            ["ollama", "pull", model_name], capture_output=True, text=True
        )

        if result.returncode == 0:
            print(f"✅ Successfully installed {model_name}")
            return True
        else:
            print(f"❌ Failed to install {model_name}")
            print(f"   Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error installing model: {e}")
        return False


def test_embedding_model(model_name="nomic-embed-text"):
    """Test the embedding model."""
    print(f"🧪 Testing embedding model: {model_name}")

    try:
        # Load local environment first
        load_local_env()

        # Add src to path for testing
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        from fintelligence_ai.rag.embeddings import get_embedding_service

        # Create embedding service
        service = get_embedding_service(model_name=model_name)

        # Test embedding generation
        test_text = "This is a test for Ollama embeddings"
        embedding = service.generate_embedding(test_text)

        print("✅ Embedding test successful!")
        print(f"   Model: {service.model_name}")
        print(f"   Local: {service.is_local}")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   Sample values: {embedding[:5]}...")

        return True

    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False


def show_installation_instructions():
    """Show instructions for installing Ollama."""
    print("📋 Ollama Installation Instructions:")
    print("=" * 50)
    print()
    print("1. Visit: https://ollama.ai/download")
    print("2. Download and install Ollama for your platform")
    print("3. After installation, run: ollama serve")
    print("4. Then re-run this script")
    print()
    print("Alternatively, install via command line:")
    print()
    print("# Linux/macOS:")
    print("curl -fsSL https://ollama.ai/install.sh | sh")
    print()
    print("# Windows:")
    print("# Download from https://ollama.ai/download")
    print()


def show_usage_instructions():
    """Show how to use the knowledge base with Ollama."""
    print("\n🎉 Setup Complete! Here's how to use it:")
    print("=" * 50)
    print()
    print("1. The knowledge base will now use Ollama embeddings by default")
    print("2. No API keys required!")
    print("3. Run the ingestion script:")
    print("   python scripts/ingest_knowledge.py")
    print()
    print("4. Check that Ollama is being used:")
    print("   Look for log messages like:")
    print("   '✅ Initialized embedding service: nomic-embed-text (local: True)'")
    print()
    print("📁 Configuration file:")
    print("   .env                       # Single environment file for all settings")


def main():
    """Main setup function."""
    print("🔧 FintelligenceAI Ollama Embeddings Setup")
    print("=" * 50)
    print()

    # Load local environment configuration
    print("📋 Loading configuration...")
    if not load_local_env():
        print("💡 Creating local configuration...")
        # This should have been created by the ingestion script setup
        print("   Run the ingestion script first to create the configuration")
    print()

    # Check if Ollama is installed
    if not check_ollama_installed():
        print("❌ Ollama is not installed")
        show_installation_instructions()
        return

    # Check if Ollama is running
    if not check_ollama_running():
        print("❌ Ollama service is not running")
        print("💡 Please start Ollama first:")
        print("   ollama serve")
        print("   Then re-run this script")
        return

    # Show current models
    print("\n📦 Current Ollama models:")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("   No models installed yet")
    except Exception as e:
        print(f"   Error listing models: {e}")

    # Install embedding model
    model_name = "nomic-embed-text"
    print(f"\n🎯 Setting up embedding model: {model_name}")

    # Check if model is already installed
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name in result.stdout:
            print(f"✅ {model_name} is already installed")
        else:
            if not install_embedding_model(model_name):
                print("❌ Failed to install embedding model")
                return
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return

    # Test the embedding service
    print()
    if test_embedding_model(model_name):
        show_usage_instructions()
    else:
        print("❌ Setup incomplete - embedding test failed")
        print("💡 Please check Ollama logs and try again")


if __name__ == "__main__":
    main()
