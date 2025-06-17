"""
Command-line interface for FintelligenceAI.

This module provides CLI commands for managing the FintelligenceAI system,
including health checks, configuration, and local mode support.
"""

import logging
from pathlib import Path

import click
import httpx
import uvicorn

from .config import get_settings

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    """FintelligenceAI CLI - Intelligent RAG Pipeline & AI Agent System."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        click.echo("🔍 Verbose mode enabled")


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Start the FintelligenceAI server."""
    click.echo("🚀 Starting FintelligenceAI server...")

    # Import here to avoid circular imports

    uvicorn.run(
        "fintelligence_ai.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@cli.command()
def health():
    """Check the health of FintelligenceAI components."""
    click.echo("🔍 Checking FintelligenceAI health...")
    settings = get_settings()

    # Check application configuration
    click.echo("\n📋 Configuration:")
    click.echo(f"  Environment: {settings.app_environment}")
    click.echo(f"  Debug Mode: {settings.app_debug}")
    click.echo(f"  Log Level: {settings.app_log_level}")
    click.echo(f"  Local Mode: {settings.dspy.local_mode}")
    click.echo(f"  Model Provider: {settings.dspy.model_provider}")

    # Check OpenAI configuration
    click.echo("\n🤖 OpenAI Configuration:")
    if settings.openai.api_key:
        click.echo("✅ OpenAI API Key: Configured")
    else:
        click.echo("⚠️  OpenAI API Key: Not configured")
    click.echo(f"  Model: {settings.openai.model}")
    click.echo(f"  Embedding Model: {settings.openai.embedding_model}")

    # Check Ollama configuration (new)
    click.echo("\n🦙 Ollama Configuration:")
    if settings.dspy.local_mode or settings.dspy.model_provider == "ollama":
        click.echo("✅ Local Mode: Enabled")
        click.echo(f"  Server URL: {settings.ollama.url}")
        click.echo(f"  Language Model: {settings.ollama.model}")
        click.echo(f"  Embedding Model: {settings.ollama.embedding_model}")

        # Test Ollama connection
        try:
            response = httpx.get(f"{settings.ollama.url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                click.echo("✅ Ollama Server: Available")

                # Check available models
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]

                # Check if required models are available
                lang_model_available = any(
                    settings.ollama.model.split(":")[0] in name for name in model_names
                )
                embed_model_available = any(
                    settings.ollama.embedding_model.split(":")[0] in name
                    for name in model_names
                )

                if lang_model_available:
                    click.echo(
                        f"✅ Language Model ({settings.ollama.model}): Available"
                    )
                else:
                    click.echo(
                        f"❌ Language Model ({settings.ollama.model}): Not found"
                    )
                    click.echo(f"   Available models: {', '.join(model_names[:5])}")

                if embed_model_available:
                    click.echo(
                        f"✅ Embedding Model ({settings.ollama.embedding_model}): Available"
                    )
                else:
                    click.echo(
                        f"❌ Embedding Model ({settings.ollama.embedding_model}): Not found"
                    )
                    click.echo(f"   Available models: {', '.join(model_names[:5])}")

            else:
                click.echo(f"❌ Ollama Server: Error (Status: {response.status_code})")

        except Exception as e:
            click.echo(f"❌ Ollama Server: Not available ({e})")
            click.echo("   Please ensure Ollama is running: ollama serve")
    else:
        click.echo("⚠️  Local Mode: Disabled")

    # Check database connections
    click.echo("\n🗄️  Database Configuration:")
    click.echo(f"  PostgreSQL URL: {settings.database.url}")
    click.echo(f"  Redis URL: {settings.redis.url}")
    click.echo(f"  ChromaDB: {settings.chromadb.host}:{settings.chromadb.port}")

    # Check file system
    click.echo("\n📁 File System:")
    paths_to_check = [
        ("Data Directory", Path(settings.files.upload_dir)),
        ("Cache Directory", Path(settings.dspy.cache_dir)),
        ("ChromaDB Persist", Path(settings.chromadb.persist_directory)),
    ]

    for name, path in paths_to_check:
        if path.exists():
            click.echo(f"✅ {name}: {path} (exists)")
        else:
            click.echo(f"⚠️  {name}: {path} (will be created)")

    click.echo("\n🎉 Health check completed!")


@cli.group()
def local():
    """Local mode management commands."""
    pass


@local.command()
def status():
    """Check local mode status and configuration."""
    settings = get_settings()

    click.echo("🦙 Local Mode Status\n")

    if not settings.dspy.local_mode and settings.dspy.model_provider != "ollama":
        click.echo("❌ Local mode is disabled")
        click.echo("   To enable: export DSPY_LOCAL_MODE=true")
        return

    click.echo("✅ Local mode is enabled")
    click.echo(f"   Provider: {settings.dspy.model_provider}")
    click.echo(f"   Ollama URL: {settings.ollama.url}")

    # Test Ollama connection
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{settings.ollama.url}/api/tags")

        if response.status_code == 200:
            click.echo("✅ Ollama server is running")

            models_data = response.json()
            models = models_data.get("models", [])

            if models:
                click.echo(f"\n📚 Available models ({len(models)}):")
                for model in models[:10]:  # Show first 10
                    name = model.get("name", "unknown")
                    size = model.get("size", 0)
                    size_gb = size / (1024**3) if size > 0 else 0
                    modified = model.get("modified_at", "")[:10]  # Date only
                    click.echo(f"   • {name} ({size_gb:.1f}GB, {modified})")

                if len(models) > 10:
                    click.echo(f"   ... and {len(models) - 10} more")
            else:
                click.echo("\n⚠️  No models found. Pull some models:")
                click.echo("   ollama pull llama3.2")
                click.echo("   ollama pull nomic-embed-text")
        else:
            click.echo(f"❌ Ollama server error: HTTP {response.status_code}")

    except Exception as e:
        click.echo(f"❌ Cannot connect to Ollama server: {e}")
        click.echo("   Make sure Ollama is running: ollama serve")


@local.command()
@click.argument("model_name")
def pull(model_name: str):
    """Pull a model for local use."""
    settings = get_settings()

    click.echo(f"📥 Pulling model: {model_name}")

    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(
                f"{settings.ollama.url}/api/pull",
                json={"name": model_name},
                timeout=300.0,
            )

        if response.status_code == 200:
            click.echo(f"✅ Successfully pulled {model_name}")
        else:
            click.echo(f"❌ Failed to pull {model_name}: HTTP {response.status_code}")
            click.echo(f"   Response: {response.text}")

    except Exception as e:
        click.echo(f"❌ Error pulling model: {e}")


@local.command()
@click.argument("model_name")
def remove(model_name: str):
    """Remove a local model."""
    settings = get_settings()

    click.echo(f"🗑️  Removing model: {model_name}")

    if not click.confirm(f"Are you sure you want to remove {model_name}?"):
        click.echo("Cancelled.")
        return

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.request(
                "DELETE", f"{settings.ollama.url}/api/delete", json={"name": model_name}
            )

        if response.status_code == 200:
            click.echo(f"✅ Successfully removed {model_name}")
        else:
            click.echo(f"❌ Failed to remove {model_name}: HTTP {response.status_code}")
            click.echo(f"   Response: {response.text}")

    except Exception as e:
        click.echo(f"❌ Error removing model: {e}")


@local.command()
def setup():
    """Setup local mode with recommended models."""
    settings = get_settings()

    click.echo("🦙 Setting up local mode for FintelligenceAI\n")

    # Check if Ollama is available
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{settings.ollama.url}/api/tags")

        if response.status_code != 200:
            click.echo(f"❌ Ollama server error: HTTP {response.status_code}")
            return

    except Exception as e:
        click.echo(f"❌ Cannot connect to Ollama server: {e}")
        click.echo("\nPlease ensure Ollama is running:")
        click.echo("   1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        click.echo("   2. Start server: ollama serve")
        return

    click.echo("✅ Ollama server is available")

    # Recommended models for different use cases
    model_options = {
        "development": {
            "language": "llama3.2:1b",
            "embedding": "nomic-embed-text",
            "description": "Fast, lightweight (4GB RAM)",
        },
        "balanced": {
            "language": "llama3.2",
            "embedding": "nomic-embed-text",
            "description": "Good balance (6GB RAM)",
        },
        "quality": {
            "language": "llama3.1:8b",
            "embedding": "mxbai-embed-large",
            "description": "High quality (12GB RAM)",
        },
    }

    click.echo("\n📚 Available model configurations:")
    for key, config in model_options.items():
        click.echo(
            f"   {key}: {config['language']} + {config['embedding']} - {config['description']}"
        )

    choice = click.prompt(
        "\nSelect configuration",
        type=click.Choice(list(model_options.keys())),
        default="balanced",
    )

    selected = model_options[choice]

    click.echo(f"\n📥 Pulling models for '{choice}' configuration...")

    # Pull language model
    click.echo(f"Pulling language model: {selected['language']}")
    try:
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{settings.ollama.url}/api/pull", json={"name": selected["language"]}
            )
        if response.status_code == 200:
            click.echo("✅ Language model pulled successfully")
        else:
            click.echo(f"❌ Failed to pull language model: {response.text}")
            return
    except Exception as e:
        click.echo(f"❌ Error pulling language model: {e}")
        return

    # Pull embedding model
    click.echo(f"Pulling embedding model: {selected['embedding']}")
    try:
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{settings.ollama.url}/api/pull", json={"name": selected["embedding"]}
            )
        if response.status_code == 200:
            click.echo("✅ Embedding model pulled successfully")
        else:
            click.echo(f"❌ Failed to pull embedding model: {response.text}")
            return
    except Exception as e:
        click.echo(f"❌ Error pulling embedding model: {e}")
        return

    # Generate environment configuration
    click.echo("\n⚙️  Configuration for .env file:")
    click.echo("# Local mode configuration")
    click.echo("DSPY_LOCAL_MODE=true")
    click.echo("DSPY_MODEL_PROVIDER=ollama")
    click.echo(f"OLLAMA_MODEL={selected['language']}")
    click.echo(f"OLLAMA_EMBEDDING_MODEL={selected['embedding']}")
    click.echo("OLLAMA_TEMPERATURE=0.1")
    click.echo("OLLAMA_MAX_TOKENS=4096")

    if click.confirm("\nSave this configuration to .env file?"):
        env_content = f"""# Local mode configuration
DSPY_LOCAL_MODE=true
DSPY_MODEL_PROVIDER=ollama
OLLAMA_MODEL={selected['language']}
OLLAMA_EMBEDDING_MODEL={selected['embedding']}
OLLAMA_TEMPERATURE=0.1
OLLAMA_MAX_TOKENS=4096
"""

        env_path = Path(".env")
        if env_path.exists():
            # Append to existing file
            with open(env_path, "a") as f:
                f.write("\n" + env_content)
            click.echo("✅ Configuration appended to .env file")
        else:
            # Create new file
            with open(env_path, "w") as f:
                f.write(env_content)
            click.echo("✅ Configuration saved to .env file")

    click.echo("\n🎉 Local mode setup completed!")
    click.echo("   Start the application: fintelligence serve")
    click.echo("   Or with Docker: docker-compose -f docker-compose.local.yml up")


@cli.command()
def init():
    """Initialize FintelligenceAI project structure."""
    click.echo("🏗️  Initializing FintelligenceAI project structure...")

    # Create necessary directories
    directories = [
        "data/uploads",
        "data/chroma",
        "data/dspy_cache",
        "data/experiments",
        "config",
        "logs",
    ]

    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            click.echo(f"✅ Created directory: {directory}")
        else:
            click.echo(f"⚠️  Directory already exists: {directory}")

    # Create default configuration files
    click.echo("\n📋 Creating default configuration...")

    # Create .env template if it doesn't exist
    env_template = Path("env.template")
    env_file = Path(".env")

    if env_template.exists() and not env_file.exists():
        import shutil

        shutil.copy(env_template, env_file)
        click.echo("✅ Created .env from template")
    elif not env_file.exists():
        click.echo("⚠️  No .env template found, please create .env manually")

    click.echo("\n🎉 Project initialization completed!")
    click.echo("   Next steps:")
    click.echo("   1. Configure .env file with your settings")
    click.echo("   2. For local mode: fintelligence local setup")
    click.echo("   3. Start the server: fintelligence serve")


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
