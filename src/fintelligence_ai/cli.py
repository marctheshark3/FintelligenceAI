"""
Command Line Interface for FintelligenceAI.

This module provides CLI commands for managing the FintelligenceAI application.
"""

import asyncio
import logging
import sys

import click
import uvicorn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from fintelligence_ai.config import get_settings

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="FintelligenceAI")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool):
    """FintelligenceAI - Intelligent RAG Pipeline & AI Agent System."""
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind the server to")
@click.option("--port", default=8000, help="Port to bind the server to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--workers", default=1, help="Number of worker processes")
def serve(host: str, port: int, reload: bool, workers: int):
    """Start the FintelligenceAI API server."""
    settings = get_settings()

    click.echo(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    click.echo(f"📍 Server will be available at: http://{host}:{port}")
    click.echo(f"📚 API documentation: http://{host}:{port}/docs")
    click.echo(f"🔧 Environment: {settings.app_environment}")

    uvicorn.run(
        "fintelligence_ai.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=settings.app_log_level.lower(),
    )


@main.command()
def info():
    """Display application information."""
    settings = get_settings()

    click.echo("=" * 50)
    click.echo(f"🤖 {settings.app_name}")
    click.echo("=" * 50)
    click.echo(f"Version: {settings.app_version}")
    click.echo(f"Environment: {settings.app_environment}")
    click.echo(f"Debug: {settings.app_debug}")
    click.echo(f"Log Level: {settings.app_log_level}")
    click.echo()
    click.echo("🔧 Configuration:")
    click.echo(f"  • API Host: {settings.api.host}:{settings.api.port}")
    click.echo(f"  • Database: {settings.database.url}")
    click.echo(f"  • Redis: {settings.redis.url}")
    click.echo(f"  • ChromaDB: {settings.chromadb.host}:{settings.chromadb.port}")
    click.echo(f"  • DSPy Cache: {settings.dspy.cache_dir}")
    click.echo()
    click.echo("🚀 Available commands:")
    click.echo("  • fintelligence serve  - Start the API server")
    click.echo("  • fintelligence info   - Display this information")


@main.command()
@click.option("--check-deps", is_flag=True, help="Check core dependencies")
def health(check_deps: bool):
    """Check application health and dependencies."""
    settings = get_settings()

    click.echo(f"🏥 Health Check for {settings.app_name}")
    click.echo("=" * 40)

    # Check basic imports
    try:
        import dspy

        click.echo("✅ DSPy: Available")
    except ImportError:
        click.echo("❌ DSPy: Not available")

    try:
        import chromadb

        click.echo("✅ ChromaDB: Available")
    except ImportError:
        click.echo("❌ ChromaDB: Not available")

    try:
        import fastapi

        click.echo("✅ FastAPI: Available")
    except ImportError:
        click.echo("❌ FastAPI: Not available")

    try:
        import langchain

        click.echo("✅ LangChain: Available")
    except ImportError:
        click.echo("❌ LangChain: Not available")

    # Check configuration
    click.echo()
    click.echo("⚙️  Configuration:")
    click.echo(f"✅ Settings loaded: {settings.app_name}")

    if settings.openai.api_key:
        click.echo("✅ OpenAI API Key: Configured")
    else:
        click.echo("⚠️  OpenAI API Key: Not configured")

    click.echo()
    click.echo("🎯 System Status: Ready for development!")


@main.group()
def knowledge():
    """Knowledge base management commands"""
    pass


@knowledge.command()
@click.option(
    "--force", "-f", is_flag=True, help="Force refresh existing knowledge base"
)
def setup(force: bool):
    """Set up the ErgoScript knowledge base from GitHub repository"""

    async def _setup():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Setting up ErgoScript knowledge base...", total=None
            )

            try:
                from fintelligence_ai.knowledge import setup_ergoscript_knowledge_base

                result = await setup_ergoscript_knowledge_base()

                if result.success:
                    progress.update(
                        task, description="✅ Knowledge base setup completed!"
                    )

                    # Display results table
                    table = Table(title="Knowledge Base Setup Results")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="green")

                    table.add_row(
                        "Documents Processed", str(result.documents_processed)
                    )
                    table.add_row("Chunks Created", str(result.chunks_created))
                    table.add_row("Chunks Stored", str(result.chunks_stored))
                    table.add_row(
                        "Processing Time", f"{result.processing_time_seconds:.2f}s"
                    )
                    table.add_row("Storage Time", f"{result.storage_time_seconds:.2f}s")

                    console.print(table)

                    if result.errors:
                        console.print("\n[yellow]Warnings/Errors:[/yellow]")
                        for error in result.errors:
                            console.print(f"  • {error}")
                else:
                    progress.update(task, description="❌ Knowledge base setup failed!")
                    console.print("[red]Errors:[/red]")
                    for error in result.errors:
                        console.print(f"  • {error}")
                    sys.exit(1)

            except Exception as e:
                progress.update(task, description="❌ Setup failed!")
                console.print(f"[red]Error: {e}[/red]")
                sys.exit(1)

    asyncio.run(_setup())


@knowledge.command()
def stats():
    """Show knowledge base statistics"""

    async def _stats():
        try:
            from fintelligence_ai.knowledge import get_knowledge_base_stats

            stats = await get_knowledge_base_stats()

            if not stats:
                console.print(
                    "[yellow]No knowledge base found. Run 'setup' first.[/yellow]"
                )
                return

            table = Table(title="Knowledge Base Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for key, value in stats.items():
                if key == "last_updated" and value:
                    table.add_row(key.replace("_", " ").title(), str(value)[:19])
                else:
                    table.add_row(key.replace("_", " ").title(), str(value))

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error getting stats: {e}[/red]")

    asyncio.run(_stats())


@knowledge.command()
@click.argument("query", required=True)
@click.option("--limit", "-l", default=5, help="Number of results to return")
def search(query: str, limit: int):
    """Search the knowledge base"""

    async def _search():
        try:
            from fintelligence_ai.knowledge.ingestion import KnowledgeBaseManager

            manager = KnowledgeBaseManager()
            await manager.initialize()

            results = await manager.search_knowledge_base(query, limit=limit)

            if not results:
                console.print("[yellow]No results found.[/yellow]")
                return

            console.print(f"\n[cyan]Search Results for: '{query}'[/cyan]\n")

            for i, result in enumerate(results, 1):
                console.print(f"[bold]Result {i}:[/bold]")
                console.print(f"Score: {result['score']:.3f}")

                # Handle metadata properly - it could be a dict or DocumentMetadata object
                metadata = result["metadata"]
                if hasattr(metadata, "dict"):
                    # It's a DocumentMetadata object
                    title = getattr(metadata, "title", None) or "Unknown"
                    source = (
                        metadata.source.value
                        if hasattr(metadata.source, "value")
                        else str(metadata.source)
                    )
                    category = (
                        metadata.category.value
                        if hasattr(metadata.category, "value")
                        else str(metadata.category)
                    )
                    complexity = (
                        metadata.complexity.value
                        if hasattr(metadata.complexity, "value")
                        else str(metadata.complexity)
                    )
                else:
                    # It's a dictionary
                    title = metadata.get("title", "Unknown")
                    source = metadata.get("source", "Unknown")
                    category = metadata.get("category", "Unknown")
                    complexity = metadata.get("complexity", "Unknown")

                console.print(f"Title: {title}")
                console.print(
                    f"Source: {source} | Category: {category} | Complexity: {complexity}"
                )
                console.print(f"Content: {result['content'][:200]}...")
                console.print("─" * 50)

        except Exception as e:
            console.print(f"[red]Error searching: {e}[/red]")

    asyncio.run(_search())


@main.group()
def rag():
    """RAG pipeline management commands"""
    pass


@rag.command()
@click.argument("query", required=True)
@click.option(
    "--strategy",
    default="hybrid",
    help="Retrieval strategy: semantic, keyword, or hybrid",
)
def test(query: str, strategy: str):
    """Test the RAG pipeline with a query"""

    async def _test():
        try:
            console.print(f"[cyan]Testing RAG pipeline with query: '{query}'[/cyan]")
            console.print(f"Strategy: {strategy}\n")

            # Create RAG pipeline
            from fintelligence_ai.rag import create_ergoscript_pipeline

            pipeline = create_ergoscript_pipeline(
                vector_store={"collection_name": "ergoscript_examples"},
                retrieval={"similarity_threshold": 0.4},  # Lower threshold for testing
            )

            # Process query
            result = pipeline.query(query_text=query, generation_type="general")

            # Display results
            console.print("[bold]Generated Response:[/bold]")
            console.print(result.generation_result.generated_text)

            console.print(
                f"\n[bold]Retrieved Documents ({len(result.retrieval_results)}):[/bold]"
            )
            for i, doc in enumerate(result.retrieval_results[:3], 1):
                title = (
                    doc.title or doc.metadata.title
                    if hasattr(doc.metadata, "title")
                    else "Unknown"
                )
                console.print(f"{i}. {title} (Score: {doc.score:.3f})")

            console.print("\n[bold]Generation Stats:[/bold]")
            console.print(
                f"Confidence: {result.generation_result.confidence_score:.3f}"
            )
            console.print(f"Processing time: {result.processing_time_ms}ms")

        except Exception as e:
            console.print(f"[red]Error testing RAG pipeline: {e}[/red]")

    asyncio.run(_test())


if __name__ == "__main__":
    main()
