#!/usr/bin/env python3
"""
FintelligenceAI Knowledge Base Ingestion Script

This script provides an easy-to-use interface for ingesting various types of content
into the FintelligenceAI knowledge base. It processes files, URLs, and GitHub repositories
from the knowledge-base folder structure.

Usage:
    python scripts/ingest_knowledge.py [options]

Examples:
    # Ingest all new content
    python scripts/ingest_knowledge.py

    # Ingest specific folder
    python scripts/ingest_knowledge.py --folder categories/tutorials

    # Dry run to preview changes
    python scripts/ingest_knowledge.py --dry-run

    # Force re-ingestion of processed files
    python scripts/ingest_knowledge.py --force
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

# Add the src directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Load local environment configuration if available
def load_local_env():
    """Load environment configuration for GitHub tokens and AI provider settings."""
    project_root = Path(__file__).parent.parent

    # Load main .env file (for all configuration including AI providers, GitHub tokens, etc.)
    main_env_path = project_root / ".env"
    if main_env_path.exists():
        try:
            import dotenv

            dotenv.load_dotenv(main_env_path)
            print(f"✅ Loaded configuration from {main_env_path}")
            print(f"Using collection: {os.getenv('CHROMA_COLLECTION_NAME')}")
        except ImportError:
            # If python-dotenv is not available, manually parse the file
            with open(main_env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        # Remove inline comments
                        value = value.split("#")[0].strip()
                        os.environ[key] = value
            print(f"✅ Loaded configuration manually from {main_env_path}")
    else:
        print(f"⚠️  No .env file found at {main_env_path}")
        print("   Please copy env.template to .env and configure your settings")


load_local_env()

from fintelligence_ai.knowledge import (
    DocumentProcessor,
    GitHubDataCollector,
    KnowledgeBaseManager,
)
from fintelligence_ai.rag.models import Document, DocumentMetadata


class ProgressTracker:
    """Enhanced progress tracking with visual feedback."""

    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = time.time()
        self.description = description
        self.current_item = ""
        self.last_update_time = time.time()

    def update(self, current_item: str = "", increment: int = 1):
        """Update progress with current item being processed."""
        self.processed_items += increment
        self.current_item = current_item
        current_time = time.time()

        # Only update display every 0.1 seconds to avoid excessive output
        if (
            current_time - self.last_update_time >= 0.1
            or self.processed_items == self.total_items
        ):
            self._display_progress()
            self.last_update_time = current_time

    def _display_progress(self):
        """Display progress bar and stats."""
        if self.total_items == 0:
            percentage = 100
        else:
            percentage = (self.processed_items / self.total_items) * 100

        # Calculate time estimates
        elapsed_time = time.time() - self.start_time
        if self.processed_items > 0:
            avg_time_per_item = elapsed_time / self.processed_items
            remaining_items = self.total_items - self.processed_items
            eta_seconds = remaining_items * avg_time_per_item

            # Format ETA
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f}m"
            else:
                eta_str = f"{eta_seconds/3600:.1f}h"

            # Calculate processing rate
            items_per_second = self.processed_items / elapsed_time
            if items_per_second >= 1:
                rate_str = f"{items_per_second:.1f} items/s"
            else:
                rate_str = f"{60/items_per_second:.1f}s/item"
        else:
            eta_str = "calculating..."
            rate_str = "calculating..."

        # Create progress bar
        bar_width = 30
        filled_width = int(bar_width * percentage / 100)
        bar = "█" * filled_width + "░" * (bar_width - filled_width)

        # Truncate current item if too long
        display_item = self.current_item
        if len(display_item) > 40:
            display_item = display_item[:37] + "..."

        # Print progress line (overwrite previous)
        progress_line = f"\r{self.description}: [{bar}] {percentage:6.1f}% ({self.processed_items}/{self.total_items}) | ETA: {eta_str} | {rate_str}"
        if display_item:
            progress_line += f" | Current: {display_item}"

        print(progress_line, end="", flush=True)

        # Print newline when complete
        if self.processed_items >= self.total_items:
            print()  # New line after completion


class IngestionConfig(BaseModel):
    """Configuration for knowledge ingestion."""

    default_category: str = "general"
    default_difficulty: str = "intermediate"
    file_patterns: dict[str, list[str]] = {}
    auto_categorize: bool = True
    extract_metadata: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    supported_extensions: list[str] = [
        ".md",
        ".pdf",
        ".txt",
        ".docx",
        ".py",
        ".es",
        ".js",
        ".json",
        ".csv",
        ".html",
    ]
    exclude_patterns: list[str] = [
        "**/node_modules/**",
        "**/.git/**",
        "**/dist/**",
        "**/build/**",
    ]
    github: dict = {}
    web_scraping: dict = {}


class IngestionStats(BaseModel):
    """Statistics for ingestion process."""

    files_processed: int = 0
    urls_processed: int = 0
    repos_processed: int = 0
    documents_created: int = 0
    errors: list[str] = []
    start_time: datetime
    end_time: Optional[datetime] = None


class KnowledgeIngestionOrchestrator:
    """Orchestrates the ingestion of knowledge from various sources."""

    def __init__(self, knowledge_base_dir: Path, config_path: Optional[Path] = None):
        self.knowledge_base_dir = knowledge_base_dir
        self.processed_dir = knowledge_base_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)

        # Load configuration
        if config_path and config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
            self.config = IngestionConfig(**config_data)
        else:
            default_config_path = knowledge_base_dir / ".knowledge-config.json"
            if default_config_path.exists():
                with open(default_config_path) as f:
                    config_data = json.load(f)
                self.config = IngestionConfig(**config_data)
            else:
                self.config = IngestionConfig()

        # Initialize components
        self.knowledge_manager = KnowledgeBaseManager()
        self.processor = DocumentProcessor()
        self.github_collector = GitHubDataCollector()

        # Set up logging
        self._setup_logging()

        # Get the actual collection name being used (provider-specific)
        collection_name = self.knowledge_manager.vector_store.config.collection_name
        print(f"🔧 Using collection: '{collection_name}' (provider-specific)")

        # Track processed files to avoid duplicates - make them collection-specific
        self.processed_files_log = (
            self.processed_dir / f"processed_files_{collection_name}.txt"
        )
        self.processed_files = self._load_processed_files()

        # GitHub repos manifest for detailed tracking - make it collection-specific
        self.github_manifest_file = (
            self.processed_dir / f"github_manifest_{collection_name}.json"
        )
        self.github_manifest = self._load_github_manifest()

        # Statistics
        self.stats = IngestionStats(start_time=datetime.now())

    def _setup_logging(self):
        """Set up logging for the ingestion process."""
        log_file = self.processed_dir / "ingestion.log"

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        # Set up logger
        self.logger = logging.getLogger("knowledge_ingestion")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _load_processed_files(self) -> set[str]:
        """Load the list of already processed files."""
        if self.processed_files_log.exists():
            with open(self.processed_files_log) as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def _mark_as_processed(self, file_path: str):
        """Mark a file as processed."""
        self.processed_files.add(file_path)
        with open(self.processed_files_log, "a") as f:
            f.write(f"{file_path}\n")

    def _load_github_manifest(self) -> dict:
        """Load GitHub repositories manifest with detailed file tracking."""
        if self.github_manifest_file.exists():
            try:
                with open(self.github_manifest_file) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load GitHub manifest: {e}")
        return {}

    def _save_github_manifest(self):
        """Save GitHub repositories manifest."""
        try:
            with open(self.github_manifest_file, "w") as f:
                json.dump(self.github_manifest, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save GitHub manifest: {e}")

    def _add_repo_to_manifest(
        self, repo_url: str, repo_name: str, files_processed: list[dict]
    ):
        """Add repository information to manifest."""
        self.github_manifest[repo_url] = {
            "repo_name": repo_name,
            "processed_at": datetime.now().isoformat(),
            "files_count": len(files_processed),
            "files": files_processed,
        }
        self._save_github_manifest()

    def clear_vector_database(self):
        """Clear all documents from the vector database."""
        try:
            # Reset the specific collection being used
            self.knowledge_manager.vector_store.reset_collection()
            print(
                f"✅ Vector database collection '{self.knowledge_manager.vector_store.config.collection_name}' cleared"
            )
            self.logger.info(
                f"Vector database collection '{self.knowledge_manager.vector_store.config.collection_name}' cleared"
            )

            # Clear processed files log for this collection
            if self.processed_files_log.exists():
                self.processed_files_log.unlink()
                print(
                    f"✅ Processed files log for collection '{self.knowledge_manager.vector_store.config.collection_name}' cleared"
                )

            # Clear GitHub manifest for this collection
            if self.github_manifest_file.exists():
                self.github_manifest_file.unlink()
                print(
                    f"✅ GitHub manifest for collection '{self.knowledge_manager.vector_store.config.collection_name}' cleared"
                )

            # Reset internal state
            self.processed_files.clear()
            self.github_manifest.clear()

        except Exception as e:
            error_msg = f"Error clearing vector database: {e}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")

    def remove_repository(self, repo_url: str):
        """Remove a specific repository from the knowledge base."""
        try:
            # Remove from GitHub manifest
            if repo_url in self.github_manifest:
                repo_info = self.github_manifest[repo_url]
                del self.github_manifest[repo_url]
                self._save_github_manifest()
                print(f"✅ Removed {repo_info['repo_name']} from manifest")

            # Remove from processed files log
            if repo_url in self.processed_files:
                self.processed_files.remove(repo_url)
                # Rewrite processed files log
                with open(self.processed_files_log, "w") as f:
                    for file_path in self.processed_files:
                        f.write(f"{file_path}\n")
                print(f"✅ Removed {repo_url} from processed files log")

            # Note: Vector database entries would need to be removed based on metadata
            # This is more complex and would require querying by source metadata
            print(
                "ℹ️  Note: Vector database entries remain. For complete removal, clear entire database."
            )

        except Exception as e:
            error_msg = f"Error removing repository {repo_url}: {e}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")

    def show_detailed_manifest(self):
        """Show detailed information about processed repositories."""
        if not self.github_manifest:
            print("📭 No GitHub repositories in manifest")
            return

        print("\n🐙 GitHub Repositories Manifest:")
        print("=" * 60)

        for repo_url, repo_info in self.github_manifest.items():
            print(f"\n📁 {repo_info['repo_name']}")
            print(f"   🔗 {repo_url}")
            print(f"   📅 Processed: {repo_info['processed_at']}")
            print(f"   📊 Files: {repo_info['files_count']}")

            if "files" in repo_info and repo_info["files"]:
                print("   📋 File details:")

                # Group by category
                categories = {}
                for file_info in repo_info["files"]:
                    category = file_info.get("category", "uncategorized")
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(file_info)

                for category, files in categories.items():
                    print(f"      📂 {category.title()} ({len(files)} files):")
                    for file_info in files[:10]:  # Show first 10 files per category
                        print(
                            f"         • {file_info['name']} ({file_info.get('extension', 'unknown')})"
                        )

                    if len(files) > 10:
                        print(f"         ... and {len(files) - 10} more files")
            print("-" * 60)

    def _count_files_to_process(
        self, folder_path: Optional[Path] = None, force: bool = False
    ) -> int:
        """Count total files that will be processed."""
        if folder_path:
            docs_dir = folder_path
        else:
            docs_dir = self.knowledge_base_dir / "documents"

        if not docs_dir.exists():
            return 0

        count = 0
        for file_path in docs_dir.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.config.supported_extensions:
                continue

            relative_path = str(file_path.relative_to(self.knowledge_base_dir))
            if not force and relative_path in self.processed_files:
                continue

            count += 1

        return count

    def _count_urls_to_process(self, force: bool = False) -> int:
        """Count URLs that will be processed."""
        urls_file = self.knowledge_base_dir / "urls.txt"

        if not urls_file.exists():
            return 0

        with open(urls_file) as f:
            urls = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

        if not urls:
            return 0

        count = 0
        for url in urls:
            if not force and url in self.processed_files:
                continue
            count += 1

        return count

    def _count_repos_to_process(self, force: bool = False) -> int:
        """Count GitHub repos that will be processed."""
        repos_file = self.knowledge_base_dir / "github-repos.txt"

        if not repos_file.exists():
            return 0

        with open(repos_file) as f:
            repos = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

        if not repos:
            return 0

        count = 0
        for repo in repos:
            if not force and repo in self.processed_files:
                continue
            count += 1

        return count

    def _extract_metadata_from_filename(self, filepath: Path) -> dict:
        """Extract metadata from filename patterns."""
        filename = filepath.stem.lower()
        metadata = {}

        # Check for difficulty patterns
        if any(d in filename for d in ["beginner", "basic", "intro"]):
            metadata["difficulty"] = "beginner"
        elif any(d in filename for d in ["advanced", "expert", "complex"]):
            metadata["difficulty"] = "advanced"
        elif any(d in filename for d in ["intermediate", "medium"]):
            metadata["difficulty"] = "intermediate"

        # Check for category patterns
        for category, patterns in self.config.file_patterns.items():
            if any(pattern.replace("*", "") in filename for pattern in patterns):
                metadata["category"] = category
                break

        return metadata

    def _determine_category(self, filepath: Path, parent_dir: str = "") -> str:
        """Determine the category based on file location and content."""
        # First check parent directory
        if parent_dir:
            parent_name = Path(parent_dir).name.lower()
            if parent_name in ["tutorials", "examples", "reference", "guides"]:
                return parent_name

        # Check filename patterns
        filename_metadata = self._extract_metadata_from_filename(filepath)
        if "category" in filename_metadata:
            return filename_metadata["category"]

        # Check file extension for code files
        ext = filepath.suffix.lower()
        if ext in [".py", ".es", ".js"]:
            return "examples"
        elif ext in [".md", ".txt"] and "readme" in filepath.name.lower():
            return "reference"

        return self.config.default_category

    async def process_documents(
        self, folder_path: Optional[Path] = None, force: bool = False
    ) -> int:
        """Process documents from the documents folder or specified folder."""
        if folder_path:
            docs_dir = folder_path
        else:
            docs_dir = self.knowledge_base_dir / "documents"

        if not docs_dir.exists():
            self.logger.warning(f"Documents directory not found: {docs_dir}")
            return 0

        # Count total files to process for progress tracking
        total_files = self._count_files_to_process(folder_path, force)
        if total_files == 0:
            self.logger.info("No files to process")
            return 0

        # Initialize progress tracker
        progress = ProgressTracker(total_files, "📄 Processing files")
        processed_count = 0

        # Process all supported files recursively
        for file_path in docs_dir.rglob("*"):
            if not file_path.is_file():
                continue

            # Check if file extension is supported
            if file_path.suffix.lower() not in self.config.supported_extensions:
                continue

            # Skip if already processed (unless force is True)
            relative_path = str(file_path.relative_to(self.knowledge_base_dir))
            if not force and relative_path in self.processed_files:
                self.logger.debug(f"Skipping already processed file: {relative_path}")
                continue

            # Update progress with current file
            progress.update(current_item=file_path.name, increment=0)

            try:
                # Extract content based on file type
                content = await self._extract_file_content(file_path)
                if not content.strip():
                    self.logger.warning(f"No content extracted from {file_path}")
                    progress.update(increment=1)
                    continue

                # Determine category and metadata
                parent_dir = str(file_path.parent.relative_to(docs_dir))
                category = self._determine_category(file_path, parent_dir)

                # Create metadata
                file_metadata = self._extract_metadata_from_filename(file_path)

                doc_metadata = DocumentMetadata(
                    source="local_files",
                    category=category,
                    complexity=file_metadata.get(
                        "difficulty", self.config.default_difficulty
                    ),
                    tags=[file_path.suffix.lstrip("."), category],
                    file_path=str(file_path),
                    title=file_path.stem,
                    language=self._detect_language(file_path),
                )

                # Create document
                document = Document(
                    id=f"local_{file_path.stem}_{hash(str(file_path))}",
                    content=content,
                    metadata=doc_metadata,
                )

                # Process and ingest
                await self._ingest_document(document)

                # Move to processed folder
                processed_file_dir = self.processed_dir / "files" / category
                processed_file_dir.mkdir(parents=True, exist_ok=True)
                processed_file_path = processed_file_dir / file_path.name

                shutil.copy2(file_path, processed_file_path)
                self._mark_as_processed(relative_path)

                processed_count += 1
                self.stats.files_processed += 1

                self.logger.info(f"Processed file: {file_path.name} -> {category}")

            except Exception as e:
                error_msg = f"Error processing file {file_path}: {e}"
                self.logger.error(error_msg)
                self.stats.errors.append(error_msg)

            # Update progress after processing
            progress.update(increment=1)

        return processed_count

    async def process_category_folders(self, force: bool = False) -> int:
        """Process documents organized in category folders."""
        categories_dir = self.knowledge_base_dir / "categories"

        if not categories_dir.exists():
            self.logger.info("No categories folder found, skipping category processing")
            return 0

        # Count categories to process
        category_dirs = [d for d in categories_dir.iterdir() if d.is_dir()]
        if not category_dirs:
            self.logger.info("No category directories found")
            return 0

        total_processed = 0

        print(f"\n📂 Processing {len(category_dirs)} categories...")
        for i, category_dir in enumerate(category_dirs, 1):
            print(
                f"[{i}/{len(category_dirs)}] Processing category: {category_dir.name}"
            )
            self.logger.info(f"Processing category: {category_dir.name}")
            count = await self.process_documents(category_dir, force)
            total_processed += count

        return total_processed

    async def process_urls(self, force: bool = False) -> int:
        """Process URLs from urls.txt file."""
        urls_file = self.knowledge_base_dir / "urls.txt"

        if not urls_file.exists():
            self.logger.info("No urls.txt file found, skipping URL processing")
            return 0

        # Read URLs
        with open(urls_file) as f:
            urls = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

        if not urls:
            self.logger.info("No URLs found in urls.txt")
            return 0

        # Count URLs to process
        total_urls = self._count_urls_to_process(force)
        if total_urls == 0:
            self.logger.info("No new URLs to process")
            return 0

        # Initialize progress tracker
        progress = ProgressTracker(total_urls, "🌐 Processing URLs")
        processed_count = 0

        for url in urls:
            if not force and url in self.processed_files:
                self.logger.debug(f"Skipping already processed URL: {url}")
                continue

            # Update progress with current URL
            parsed_url = urlparse(url)
            display_url = parsed_url.netloc + parsed_url.path
            progress.update(current_item=display_url, increment=0)

            try:
                # Scrape content
                content = await self._scrape_url_content(url)
                if not content.strip():
                    self.logger.warning(f"No content scraped from {url}")
                    progress.update(increment=1)
                    continue

                # Create metadata
                title = (
                    self._extract_title_from_content(content)
                    or f"Content from {parsed_url.netloc}"
                )

                doc_metadata = DocumentMetadata(
                    source="web_scraping",
                    category="reference",
                    complexity=self.config.default_difficulty,
                    tags=["web", "scraped", parsed_url.netloc.replace(".", "_")],
                    url=url,
                    title=title,
                )

                # Create document
                document = Document(
                    id=f"url_{hash(url)}", content=content, metadata=doc_metadata
                )

                # Process and ingest
                await self._ingest_document(document)

                self._mark_as_processed(url)
                processed_count += 1
                self.stats.urls_processed += 1

                self.logger.info(f"Processed URL: {url}")

            except Exception as e:
                error_msg = f"Error processing URL {url}: {e}"
                self.logger.error(error_msg)
                self.stats.errors.append(error_msg)

            # Update progress after processing
            progress.update(increment=1)

        return processed_count

    async def process_github_repos(self, force: bool = False) -> int:
        """Process GitHub repositories using fast git clone method."""
        repos_file = self.knowledge_base_dir / "github-repos.txt"

        if not repos_file.exists():
            self.logger.info(
                "No github-repos.txt file found, skipping GitHub processing"
            )
            return 0

        # Read repositories
        with open(repos_file) as f:
            repos = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

        if not repos:
            self.logger.info("No repositories found in github-repos.txt")
            return 0

        # Count repos to process
        total_repos = self._count_repos_to_process(force)
        if total_repos == 0:
            self.logger.info("No new repositories to process")
            return 0

        # Check if git is available
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error(
                "Git is not available. Please install git to process GitHub repositories."
            )
            print(
                "❌ Git is not available. Please install git to process GitHub repositories."
            )
            return 0

        # Initialize progress tracker
        progress = ProgressTracker(total_repos, "🐙 Cloning & processing GitHub repos")
        processed_count = 0

        for repo in repos:
            if not force and repo in self.processed_files:
                self.logger.debug(f"Skipping already processed repo: {repo}")
                continue

            # Update progress with current repo
            progress.update(current_item=f"Cloning {repo}", increment=0)

            temp_dir = None
            try:
                # Parse repository URL/name
                owner, repo_name = self._parse_github_repo(repo)

                # Create temporary directory for cloning
                temp_dir = tempfile.mkdtemp(prefix=f"fintelligence_clone_{repo_name}_")
                clone_path = Path(temp_dir) / repo_name

                # Build clone URL
                if repo.startswith("https://"):
                    clone_url = repo
                else:
                    clone_url = f"https://github.com/{owner}/{repo_name}.git"

                # Clone repository with shallow clone for speed
                self.logger.info(f"Cloning repository: {clone_url}")
                progress.update(current_item=f"Cloning {repo_name}", increment=0)

                clone_result = subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--quiet",
                        clone_url,
                        str(clone_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )  # 5 minute timeout

                if clone_result.returncode != 0:
                    error_msg = f"Failed to clone {repo}: {clone_result.stderr}"
                    self.logger.error(error_msg)
                    self.stats.errors.append(error_msg)
                    progress.update(increment=1)
                    continue

                # Update progress to show processing files
                progress.update(
                    current_item=f"Processing {repo_name} files", increment=0
                )

                # Count files to process for this repo
                repo_files = []
                for file_path in clone_path.rglob("*"):
                    if not file_path.is_file():
                        continue

                    # Skip files that match exclude patterns
                    relative_path = file_path.relative_to(clone_path)
                    skip_file = False
                    for pattern in self.config.exclude_patterns:
                        if file_path.match(pattern) or str(relative_path).startswith(
                            ".git/"
                        ):
                            skip_file = True
                            break

                    if skip_file:
                        continue

                    # Check if file extension is supported
                    if file_path.suffix.lower() in self.config.supported_extensions:
                        repo_files.append(file_path)

                if not repo_files:
                    self.logger.warning(
                        f"No supported files found in repository: {repo}"
                    )
                    progress.update(increment=1)
                    continue

                # Process each file
                files_processed = 0
                processed_files_list = []

                for file_path in repo_files:
                    try:
                        # Extract content
                        content = await self._extract_file_content(file_path)
                        if not content.strip():
                            continue

                        # Calculate relative path from repo root for category determination
                        relative_path = file_path.relative_to(clone_path)
                        category = self._determine_category(
                            file_path, str(relative_path.parent)
                        )

                        # Create metadata
                        doc_metadata = DocumentMetadata(
                            source="github",
                            category=category,
                            complexity=self.config.default_difficulty,
                            tags=[
                                "github",
                                owner,
                                repo_name,
                                file_path.suffix.lstrip("."),
                            ],
                            file_path=str(relative_path),
                            url=f"https://github.com/{owner}/{repo_name}/blob/main/{relative_path}",
                            title=file_path.stem,
                            language=self._detect_language(file_path),
                        )

                        # Create document
                        document = Document(
                            id=f"github_{owner}_{repo_name}_{hash(str(relative_path))}",
                            content=content,
                            metadata=doc_metadata,
                        )

                        # Process and ingest
                        await self._ingest_document(document)
                        self.stats.documents_created += 1
                        files_processed += 1

                        # Track processed file details
                        processed_files_list.append(
                            {
                                "path": str(relative_path),
                                "name": file_path.name,
                                "extension": file_path.suffix,
                                "category": category,
                                "size_bytes": file_path.stat().st_size
                                if file_path.exists()
                                else 0,
                                "url": f"https://github.com/{owner}/{repo_name}/blob/main/{relative_path}",
                            }
                        )

                    except Exception as e:
                        error_msg = f"Error processing file {file_path}: {e}"
                        self.logger.error(error_msg)
                        self.stats.errors.append(error_msg)

                # Save to manifest
                self._add_repo_to_manifest(repo, repo_name, processed_files_list)

                self._mark_as_processed(repo)
                processed_count += 1
                self.stats.repos_processed += 1

                self.logger.info(
                    f"Processed repository: {repo} ({files_processed} files)"
                )

            except subprocess.TimeoutExpired:
                error_msg = f"Timeout cloning repository: {repo}"
                self.logger.error(error_msg)
                self.stats.errors.append(error_msg)
            except Exception as e:
                error_msg = f"Error processing repository {repo}: {e}"
                self.logger.error(error_msg)
                self.stats.errors.append(error_msg)
            finally:
                # Clean up temporary directory
                if temp_dir and Path(temp_dir).exists():
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to clean up temp directory {temp_dir}: {e}"
                        )

                # Update progress after processing
                progress.update(increment=1)

        return processed_count

    async def _extract_file_content(self, file_path: Path) -> str:
        """Extract content from various file types."""
        ext = file_path.suffix.lower()

        try:
            if ext in [".txt", ".md"]:
                with open(file_path, encoding="utf-8") as f:
                    return f.read()
            elif ext in [".py", ".es", ".js"]:
                with open(file_path, encoding="utf-8") as f:
                    return f.read()
            elif ext == ".pdf":
                # Use PyPDF2 or similar library for PDF extraction
                # For now, return placeholder
                return f"PDF content from {file_path.name} - PDF extraction not implemented yet"
            elif ext == ".json":
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2)
            else:
                # Try to read as text
                with open(file_path, encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            self.logger.error(f"Error extracting content from {file_path}: {e}")
            return ""

    async def _scrape_url_content(self, url: str) -> str:
        """Scrape content from a URL."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Basic content extraction (could be enhanced with BeautifulSoup)
                content = response.text

                # Remove HTML tags if it's HTML content
                if "text/html" in response.headers.get("content-type", ""):
                    # Basic HTML tag removal (should use proper HTML parser)
                    import re

                    content = re.sub(r"<[^>]+>", "", content)
                    content = re.sub(r"\s+", " ", content).strip()

                return content
        except Exception as e:
            self.logger.error(f"Error scraping URL {url}: {e}")
            return ""

    def _parse_github_repo(self, repo: str) -> tuple[str, str]:
        """Parse GitHub repository string to owner and repo name."""
        if repo.startswith("https://github.com/"):
            repo = repo.replace("https://github.com/", "")

        if repo.endswith(".git"):
            repo = repo[:-4]

        parts = repo.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid repository format: {repo}")

        return parts[0], parts[1]

    def _detect_language(self, file_path: Path) -> str:
        """Detect the programming language from file extension."""
        ext = file_path.suffix.lower()
        language_map = {
            ".py": "python",
            ".es": "ergoscript",
            ".js": "javascript",
            ".md": "markdown",
            ".json": "json",
            ".txt": "text",
            ".html": "html",
        }
        return language_map.get(ext, "text")

    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract title from content (look for first heading or similar)."""
        lines = content.split("\n")
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
            elif line.startswith("<h1"):
                # Basic HTML h1 extraction
                import re

                match = re.search(r"<h1[^>]*>(.*?)</h1>", line, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        return None

    async def _ingest_document(self, document: Document):
        """Ingest a single document into the knowledge base."""
        try:
            await self.knowledge_manager.initialize()

            # Add the document to the vector store
            doc_id = self.knowledge_manager.vector_store.add_document(document)
            self.logger.debug(f"Successfully ingested document: {doc_id}")

            self.stats.documents_created += 1

        except Exception as e:
            self.logger.error(f"Error ingesting document {document.id}: {e}")
            raise

    async def run_full_ingestion(
        self, folder: Optional[str] = None, force: bool = False, dry_run: bool = False
    ) -> IngestionStats:
        """Run the complete ingestion process."""
        print("\n🚀 Starting FintelligenceAI Knowledge Base Ingestion...")
        print("=" * 60)

        if dry_run:
            print("⚠️  DRY RUN MODE - No changes will be made")

        if force:
            print("🔄 FORCE MODE - Re-processing all files")

        self.logger.info("Starting knowledge base ingestion...")

        try:
            await self.knowledge_manager.initialize()

            if folder:
                # Process specific folder
                folder_path = self.knowledge_base_dir / folder
                if folder_path.exists():
                    print(f"\n📁 Processing specific folder: {folder}")
                    await self.process_documents(folder_path, force)
                else:
                    self.logger.error(f"Folder not found: {folder}")
                    print(f"❌ Folder not found: {folder}")
                    return self.stats
            else:
                # Process all sources with progress indicators
                print("\n📊 Scanning for content to process...")

                # Count total items across all sources
                total_files = self._count_files_to_process(force=force)
                total_urls = self._count_urls_to_process(force=force)
                total_repos = self._count_repos_to_process(force=force)

                total_items = total_files + total_urls + total_repos

                if total_items == 0:
                    print("✅ No new content to process!")
                    return self.stats

                print(f"Found {total_items} items to process:")
                print(f"  📄 Files: {total_files}")
                print(f"  🌐 URLs: {total_urls}")
                print(f"  🐙 GitHub repos: {total_repos}")
                print()

                # Process all sources
                if total_files > 0:
                    await self.process_documents(force=force)

                if self.knowledge_base_dir.joinpath("categories").exists():
                    await self.process_category_folders(force=force)

                if total_urls > 0:
                    await self.process_urls(force=force)

                if total_repos > 0:
                    await self.process_github_repos(force=force)

            self.stats.end_time = datetime.now()

            # Print summary
            self._print_summary()

        except Exception as e:
            error_msg = f"Ingestion failed: {e}"
            self.logger.error(error_msg)
            self.stats.errors.append(error_msg)
            print(f"\n❌ Ingestion failed: {e}")
            raise

        return self.stats

    def _print_summary(self):
        """Print ingestion summary."""
        duration = (self.stats.end_time - self.stats.start_time).total_seconds()

        print("\n" + "=" * 60)
        print("🎉 KNOWLEDGE BASE INGESTION COMPLETE!")
        print("=" * 60)

        # Format duration nicely
        if duration < 60:
            duration_str = f"{duration:.1f} seconds"
        elif duration < 3600:
            duration_str = f"{duration/60:.1f} minutes"
        else:
            duration_str = f"{duration/3600:.1f} hours"

        print(f"⏱️  Duration: {duration_str}")
        print(f"📄 Files processed: {self.stats.files_processed}")
        print(f"🌐 URLs processed: {self.stats.urls_processed}")
        print(f"🐙 Repositories processed: {self.stats.repos_processed}")
        print(f"📚 Documents created: {self.stats.documents_created}")

        # Calculate processing speed
        total_items = (
            self.stats.files_processed
            + self.stats.urls_processed
            + self.stats.repos_processed
        )
        if total_items > 0 and duration > 0:
            items_per_second = total_items / duration
            if items_per_second >= 1:
                speed_str = f"{items_per_second:.1f} items/second"
            else:
                speed_str = f"{duration/total_items:.1f} seconds/item"
            print(f"⚡ Processing speed: {speed_str}")

        if self.stats.errors:
            print(f"\n⚠️  Errors encountered: {len(self.stats.errors)}")

            # Check for rate limit errors
            rate_limit_errors = [
                e for e in self.stats.errors if "rate limit" in e.lower()
            ]
            if rate_limit_errors:
                print("\n🚨 GitHub Rate Limit Issues Detected:")
                print("   To fix this, set up a GitHub Personal Access Token:")
                print("   1. Run: python scripts/setup_github_token.py")
                print("   2. Or manually set GITHUB_TOKEN in your .env file")
                print(
                    "   3. This will increase your rate limit from 60/hour to 5,000/hour"
                )

            print("\n📋 Error details:")
            for error in self.stats.errors[:5]:  # Show first 5 errors
                print(f"  ❌ {error}")
            if len(self.stats.errors) > 5:
                print(f"  📝 ... and {len(self.stats.errors) - 5} more errors")
        else:
            print("\n✅ No errors encountered!")

        print("\n📊 Detailed logs available at:")
        print(f"   {self.processed_dir / 'ingestion.log'}")

        if total_items > 0:
            print(f"\n🎯 Successfully processed {total_items} items!")

        # Display processed files tree
        self._display_processed_tree()

        print("=" * 60)

    def _display_processed_tree(self):
        """Display a tree structure of all processed content."""
        print("\n🌳 Processed Content Tree:")
        print("─" * 40)

        # Get summary statistics
        summary = self._get_processed_content_summary()

        # Display summary
        print("📊 Summary:")
        print(f"   🐙 GitHub Repositories: {summary['total_repos']}")
        print(f"   📄 Local Files: {summary['total_files']}")
        print(f"   🌐 URLs: {summary['total_urls']}")
        print(f"   📚 Total Documents: {summary['total_documents']}")

        if summary["file_types"]:
            print(
                f"   📝 File Types: {', '.join(f'{ext}({count})' for ext, count in sorted(summary['file_types'].items()))}"
            )

        if summary["categories"]:
            print(
                f"   📂 Categories: {', '.join(f'{cat}({count})' for cat, count in sorted(summary['categories'].items()))}"
            )

        print()

        # Create tree structure from processed files log
        tree_data = self._build_tree_structure()

        if not tree_data:
            print("   (No processed content to display)")
            return

        # Display the tree
        for source_type, items in tree_data.items():
            if not items:  # Skip empty sections
                continue

            print(f"📁 {source_type}")
            for i, (item_name, details) in enumerate(items.items()):
                is_last = i == len(items) - 1
                prefix = "└── " if is_last else "├── "

                if source_type == "GitHub Repositories":
                    print(f"   {prefix}🐙 {item_name}")
                    if details["files"]:
                        for j, file_info in enumerate(details["files"]):
                            file_is_last = j == len(details["files"]) - 1
                            file_prefix = "    └── " if file_is_last else "    ├── "
                            if is_last:
                                file_prefix = "    " + file_prefix
                            else:
                                file_prefix = "│   " + file_prefix
                            print(f"   {file_prefix}📄 {file_info}")
                elif source_type == "Local Files":
                    print(f"   {prefix}📄 {item_name} → {details['category']}")
                elif source_type == "URLs":
                    print(f"   {prefix}🌐 {item_name}")

        print()

    def _build_tree_structure(self) -> dict:
        """Build a tree structure from processed files and ingestion logs."""
        tree_data = {"Local Files": {}, "URLs": {}, "GitHub Repositories": {}}

        # Parse the processed files log
        if self.processed_files_log.exists():
            with open(self.processed_files_log) as f:
                processed_items = [line.strip() for line in f if line.strip()]
        else:
            processed_items = []

        # Also look at the ingestion log for GitHub repo file details
        github_repo_files = self._extract_github_files_from_log()

        for item in processed_items:
            if item.startswith("https://github.com/"):
                # GitHub repository
                repo_name = item.split("/")[-1]
                if repo_name.endswith(".git"):
                    repo_name = repo_name[:-4]

                tree_data["GitHub Repositories"][repo_name] = {
                    "url": item,
                    "files": github_repo_files.get(item, []),
                }
            elif item.startswith("http"):
                # URL
                tree_data["URLs"][item] = {"url": item}
            else:
                # Local file
                # Determine category from file path
                if "categories/" in item:
                    category = item.split("categories/")[1].split("/")[0]
                else:
                    category = "general"

                file_name = Path(item).name
                tree_data["Local Files"][file_name] = {
                    "category": category,
                    "path": item,
                }

        # Remove empty categories
        return {k: v for k, v in tree_data.items() if v}

    def _extract_github_files_from_log(self) -> dict[str, list[str]]:
        """Extract GitHub repository file information from manifest and logs."""
        github_files = {}

        # First try to get detailed info from GitHub manifest
        for repo_url, repo_info in self.github_manifest.items():
            repo_name = repo_info.get("repo_name", "unknown")
            files = repo_info.get("files", [])
            processed_at = repo_info.get("processed_at", "unknown")

            file_list = []
            # Add summary info
            file_list.append(f"📊 {len(files)} files processed")
            file_list.append(f"📅 Last processed: {processed_at}")

            # Group files by category for better display
            categories = {}
            for file_info in files:
                category = file_info.get("category", "uncategorized")
                if category not in categories:
                    categories[category] = []
                categories[category].append(file_info["name"])

            # Add category breakdown
            for category, category_files in categories.items():
                file_list.append(f"📂 {category.title()}: {len(category_files)} files")
                # Show first few files as examples
                for file_name in category_files[:3]:
                    file_list.append(f"   • {file_name}")
                if len(category_files) > 3:
                    file_list.append(f"   • ... and {len(category_files) - 3} more")

            github_files[repo_url] = file_list

        # Fallback to log parsing if manifest is empty
        if not github_files:
            log_file = self.processed_dir / "ingestion.log"
            if log_file.exists():
                try:
                    with open(log_file) as f:
                        for line in f:
                            if "Processed repository:" in line and "files)" in line:
                                parts = line.split("Processed repository: ")[1].split(
                                    " ("
                                )
                                repo_url = parts[0]
                                file_count = parts[1].split(" files)")[0]

                                github_files[repo_url] = [
                                    f"📊 {file_count} files processed",
                                    f"📅 Last processed: {line.split()[0]} {line.split()[1]}",
                                ]
                except Exception as e:
                    self.logger.debug(f"Could not parse GitHub files from log: {e}")

        return github_files

    def _get_processed_content_summary(self) -> dict:
        """Get a summary of all processed content with statistics."""
        summary = {
            "total_repos": 0,
            "total_files": 0,
            "total_urls": 0,
            "total_documents": 0,
            "file_types": {},
            "categories": {},
        }

        # Count items from processed files log
        if self.processed_files_log.exists():
            with open(self.processed_files_log) as f:
                for line in f:
                    item = line.strip()
                    if item.startswith("https://github.com/"):
                        summary["total_repos"] += 1
                    elif item.startswith("http"):
                        summary["total_urls"] += 1
                    else:
                        summary["total_files"] += 1

                        # Count file types
                        ext = Path(item).suffix.lower()
                        if ext:
                            summary["file_types"][ext] = (
                                summary["file_types"].get(ext, 0) + 1
                            )

                        # Count categories
                        if "categories/" in item:
                            category = item.split("categories/")[1].split("/")[0]
                        else:
                            category = "general"
                        summary["categories"][category] = (
                            summary["categories"].get(category, 0) + 1
                        )

        # Count documents from log
        log_file = self.processed_dir / "ingestion.log"
        if log_file.exists():
            try:
                with open(log_file) as f:
                    content = f.read()
                    # Look for document creation entries
                    import re

                    doc_matches = re.findall(r"(\d+) files processed", content)
                    if doc_matches:
                        summary["total_documents"] = sum(
                            int(match) for match in doc_matches
                        )
            except Exception:
                pass

        return summary


def main():
    """Main entry point for the ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest knowledge into FintelligenceAI knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic ingestion
  python scripts/ingest_knowledge.py
  python scripts/ingest_knowledge.py --folder categories/tutorials
  python scripts/ingest_knowledge.py --dry-run
  python scripts/ingest_knowledge.py --force --verbose

  # Database management
  python scripts/ingest_knowledge.py --show-tree
  python scripts/ingest_knowledge.py --show-manifest
  python scripts/ingest_knowledge.py --clear-db
  python scripts/ingest_knowledge.py --remove-repo "https://github.com/microsoft/vscode-docs"
        """,
    )

    parser.add_argument(
        "--folder", help="Specific folder to process (relative to knowledge-base/)"
    )
    parser.add_argument(
        "--category", help="Override default category for all processed documents"
    )
    parser.add_argument("--config", type=Path, help="Path to custom configuration file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be processed without making changes",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-process already processed files"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--show-tree",
        action="store_true",
        help="Show tree structure of processed content and exit",
    )

    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear the entire vector database and processed files",
    )

    parser.add_argument(
        "--remove-repo",
        help="Remove a specific GitHub repository from tracking (provide repo URL)",
    )

    parser.add_argument(
        "--show-manifest",
        action="store_true",
        help="Show detailed GitHub repositories manifest and exit",
    )

    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Find knowledge-base directory
    script_dir = Path(__file__).parent
    knowledge_base_dir = script_dir.parent / "knowledge-base"

    if not knowledge_base_dir.exists():
        print(f"Error: knowledge-base directory not found at {knowledge_base_dir}")
        print("Please run this script from the project root directory.")
        sys.exit(1)

    # Create orchestrator
    orchestrator = KnowledgeIngestionOrchestrator(
        knowledge_base_dir=knowledge_base_dir, config_path=args.config
    )

    # Handle special commands that don't require ingestion
    if args.show_tree:
        print("📊 FintelligenceAI Knowledge Base - Processed Content")
        print("=" * 60)
        orchestrator._display_processed_tree()
        return

    if args.show_manifest:
        orchestrator.show_detailed_manifest()
        return

    if args.clear_db:
        print("⚠️  This will clear the entire vector database and all processed files.")
        confirm = input("Are you sure? Type 'yes' to confirm: ")
        if confirm.lower() == "yes":
            orchestrator.clear_vector_database()
            print("✅ Database cleared successfully")
        else:
            print("❌ Operation cancelled")
        return

    if args.remove_repo:
        orchestrator.remove_repository(args.remove_repo)
        return

    # Override category if specified
    if args.category:
        orchestrator.config.default_category = args.category

    # Run ingestion
    try:
        stats = asyncio.run(
            orchestrator.run_full_ingestion(
                folder=args.folder, force=args.force, dry_run=args.dry_run
            )
        )

        if stats.errors:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nIngestion cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
