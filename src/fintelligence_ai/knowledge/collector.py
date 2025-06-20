"""
Data Collection Module

Collects ErgoScript examples and documentation from various sources.
"""

import asyncio
import logging
import os
import re
import time
from functools import wraps
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel

from ..config import get_settings
from ..rag.models import Document

logger = logging.getLogger(__name__)
settings = get_settings()


def rate_limit(calls_per_hour: int = 5000):
    """Decorator to enforce rate limiting on GitHub API calls."""
    min_interval = 3600 / calls_per_hour  # Minimum interval between calls in seconds

    def decorator(func):
        last_called = [0.0]

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Calculate time since last call
            now = time.time()
            time_passed = now - last_called[0]

            # If we need to wait, do so
            if time_passed < min_interval:
                wait_time = min_interval - time_passed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)

            last_called[0] = time.time()
            return await func(*args, **kwargs)

        return wrapper

    return decorator


async def retry_with_exponential_backoff(
    func, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0
):
    """Retry function with exponential backoff for rate limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403 and "rate limit" in str(e).lower():
                if attempt == max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded for rate limit")
                    raise

                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    f"Rate limit hit, retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries + 1})"
                )
                await asyncio.sleep(delay)
            else:
                raise
        except Exception:
            raise


class GitHubFile(BaseModel):
    """Represents a file from GitHub repository."""

    name: str
    path: str
    download_url: str
    content: Optional[str] = None
    metadata: dict = {}


class GitHubDataCollector:
    """Collects data from GitHub repositories with rate limiting and authentication."""

    def __init__(self):
        self.session: Optional[httpx.AsyncClient] = None
        self.base_url = "https://api.github.com"
        self.github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_API_KEY")

        # Set rate limits based on authentication
        if self.github_token:
            self.rate_limit_per_hour = 5000  # Authenticated
            logger.info("Using authenticated GitHub API requests (5,000 requests/hour)")
        else:
            self.rate_limit_per_hour = 60  # Unauthenticated
            logger.warning(
                "Using unauthenticated GitHub API requests (60 requests/hour). Consider setting GITHUB_TOKEN for higher limits."
            )

    async def __aenter__(self):
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "FintelligenceAI-KnowledgeCollector/1.0",
        }

        # Add authentication if token is available
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        self.session = httpx.AsyncClient(timeout=30.0, headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    async def _make_api_request(self, url: str) -> httpx.Response:
        """Make a rate-limited API request with retry logic."""
        if not self.session:
            raise RuntimeError(
                "GitHubDataCollector must be used as async context manager"
            )

        # Apply rate limiting based on authentication status
        if not hasattr(self, "_last_request_time"):
            self._last_request_time = 0.0

        # Calculate minimum interval between requests
        min_interval = 3600 / self.rate_limit_per_hour
        now = time.time()
        time_since_last = now - self._last_request_time

        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)

        async def _request():
            response = await self.session.get(url)
            response.raise_for_status()
            return response

        result = await retry_with_exponential_backoff(_request)
        self._last_request_time = time.time()
        return result

    async def get_repository_files(
        self,
        owner: str,
        repo: str,
        path: str = "",
        file_extensions: Optional[set[str]] = None,
    ) -> list[GitHubFile]:
        """Get files from a GitHub repository with rate limiting."""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
            response = await self._make_api_request(url)
            contents = response.json()
            files = []

            for item in contents:
                if item["type"] == "file":
                    # Filter by file extensions if specified
                    if file_extensions:
                        file_ext = Path(item["name"]).suffix.lower()
                        if file_ext not in file_extensions:
                            continue

                    files.append(
                        GitHubFile(
                            name=item["name"],
                            path=item["path"],
                            download_url=item["download_url"],
                            metadata={
                                "size": item["size"],
                                "sha": item["sha"],
                                "url": item["html_url"],
                            },
                        )
                    )
                elif item["type"] == "dir":
                    # Recursively get files from subdirectories
                    # Add small delay to avoid overwhelming API
                    await asyncio.sleep(0.1)
                    subdir_files = await self.get_repository_files(
                        owner, repo, item["path"], file_extensions
                    )
                    files.extend(subdir_files)

            return files

        except Exception as e:
            logger.error(f"Error fetching repository files: {e}")
            return []

    async def download_file_content(self, file: GitHubFile) -> str:
        """Download content of a specific file."""
        if not self.session:
            raise RuntimeError(
                "GitHubDataCollector must be used as async context manager"
            )

        try:
            # Use the download_url which doesn't count against API rate limits
            response = await self.session.get(file.download_url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error downloading file {file.name}: {e}")
            return ""

    async def check_rate_limit_status(self) -> dict:
        """Check current rate limit status."""
        try:
            url = f"{self.base_url}/rate_limit"
            response = await self._make_api_request(url)
            return response.json()
        except Exception as e:
            logger.error(f"Error checking rate limit status: {e}")
            return {}


class ErgoScriptCollector:
    """Specialized collector for ErgoScript examples and documentation."""

    def __init__(self):
        self.github_collector = GitHubDataCollector()

        # ErgoScript by Example repository
        self.repo_owner = "ergoplatform"
        self.repo_name = "ergoscript-by-example"

        # Difficulty mapping
        self.difficulty_map = {
            "starter": 1,
            "beginner": 2,
            "intermediate": 3,
            "expert": 4,
        }

    async def collect_ergoscript_examples(self) -> list[Document]:
        """Collect all ErgoScript examples from the repository."""
        documents = []

        async with self.github_collector:
            # Get all markdown files
            files = await self.github_collector.get_repository_files(
                self.repo_owner, self.repo_name, file_extensions={".md"}
            )

            # Filter out README and template files
            example_files = [
                f
                for f in files
                if f.name.lower() not in ["readme.md", "example_template.md"]
                and not f.name.startswith(".")
            ]

            # Process each example file
            for file in example_files:
                content = await self.github_collector.download_file_content(file)
                if content:
                    document = await self._process_ergoscript_example(file, content)
                    if document:
                        documents.append(document)

        logger.info(f"Collected {len(documents)} ErgoScript examples")
        return documents

    async def _process_ergoscript_example(
        self, file: GitHubFile, content: str
    ) -> Optional[Document]:
        """Process a single ErgoScript example file."""
        try:
            # Extract metadata from content
            metadata = self._extract_metadata(file, content)

            # Extract code blocks
            _ = self._extract_code_blocks(content)

            # Clean and structure content
            cleaned_content = self._clean_content(content)

            # Create DocumentMetadata with proper enum values
            from ..rag.models import DocumentMetadata

            # Map difficulty to complexity enum
            complexity_mapping = {
                "starter": "beginner",
                "beginner": "beginner",
                "intermediate": "intermediate",
                "expert": "advanced",
            }

            doc_metadata = DocumentMetadata(
                source="examples",  # Valid enum value
                category="examples",  # Valid enum value - these are ErgoScript examples
                complexity=complexity_mapping.get(
                    metadata["difficulty"].lower(), "beginner"
                ),
                tags=["ergoscript"] + metadata.get("tags", []),
                file_path=file.path,
                url=file.metadata.get("url", ""),
                language="ergoscript",
                tested=False,
            )

            document = Document(
                id=f"ergoscript_example_{file.name.replace('.md', '')}",
                content=cleaned_content,
                title=metadata["title"],
                metadata=doc_metadata,
            )

            return document

        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
            return None

    def _extract_metadata(self, file: GitHubFile, content: str) -> dict:
        """Extract metadata from the example content."""
        metadata = {
            "title": self._extract_title(content)
            or file.name.replace(".md", "").replace("_", " ").title(),
            "difficulty": self._extract_difficulty(content),
            "tags": self._extract_tags(content),
            "use_cases": self._extract_use_cases(content),
        }

        # Add difficulty score for sorting
        difficulty_text = metadata["difficulty"].lower()
        metadata["difficulty_score"] = self.difficulty_map.get(difficulty_text, 2)

        return metadata

    def _extract_title(self, content: str) -> Optional[str]:
        """Extract title from markdown content."""
        # Look for main heading
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            return title_match.group(1).strip()
        return None

    def _extract_difficulty(self, content: str) -> str:
        """Extract difficulty level from content."""
        # Look for difficulty indicators in content
        content_lower = content.lower()

        if "expert" in content_lower:
            return "Expert"
        elif "intermediate" in content_lower:
            return "Intermediate"
        elif "beginner" in content_lower:
            return "Beginner"
        elif "starter" in content_lower:
            return "Starter"

        # Default to beginner if not specified
        return "Beginner"

    def _extract_tags(self, content: str) -> list[str]:
        """Extract relevant tags from content."""
        tags = []
        content_lower = content.lower()

        # Common ErgoScript patterns
        tag_patterns = {
            "token": ["token", "erc-20", "asset"],
            "contract": ["contract", "smart contract"],
            "swap": ["swap", "exchange", "trade"],
            "lock": ["lock", "timelock", "pin"],
            "escrow": ["escrow", "deposit"],
            "game": ["game", "gambling", "heads", "tails"],
            "oracle": ["oracle", "data feed"],
            "dex": ["dex", "decentralized exchange"],
            "auction": ["auction", "bid"],
            "stealth": ["stealth", "privacy"],
            "multisig": ["multisig", "multi-signature"],
        }

        for tag, patterns in tag_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                tags.append(tag)

        return tags

    def _extract_use_cases(self, content: str) -> list[str]:
        """Extract use cases from content."""
        use_cases = []
        content_lower = content.lower()

        use_case_patterns = {
            "DeFi": ["defi", "decentralized finance", "yield", "liquidity"],
            "Gaming": ["game", "gambling", "random", "chance"],
            "Privacy": ["privacy", "stealth", "anonymous"],
            "Security": ["security", "lock", "escrow", "multisig"],
            "Trading": ["trading", "swap", "exchange", "market"],
            "Governance": ["governance", "voting", "dao"],
            "Identity": ["identity", "authentication", "verification"],
        }

        for use_case, patterns in use_case_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                use_cases.append(use_case)

        return use_cases

    def _extract_code_blocks(self, content: str) -> list[dict]:
        """Extract code blocks from markdown content."""
        code_blocks = []

        # Find all code blocks with language specification
        code_pattern = r"```(\w*)\n(.*?)\n```"
        matches = re.findall(code_pattern, content, re.DOTALL)

        for i, (language, code) in enumerate(matches):
            code_blocks.append(
                {
                    "language": language or "scala",
                    "code": code.strip(),
                    "block_number": i + 1,
                }
            )

        return code_blocks

    def _clean_content(self, content: str) -> str:
        """Clean and structure the content for better retrieval."""
        # Remove excessive whitespace
        content = re.sub(r"\n\s*\n", "\n\n", content)

        # Remove markdown artifacts that might interfere with retrieval
        content = re.sub(r"^\[.*?\]:\s*.*$", "", content, flags=re.MULTILINE)

        # Ensure proper spacing around headers
        content = re.sub(r"^(#{1,6})\s*(.+)$", r"\1 \2", content, flags=re.MULTILINE)

        return content.strip()


# Convenience function for easy collection
async def collect_ergoscript_knowledge_base() -> list[Document]:
    """Collect the complete ErgoScript knowledge base."""
    collector = ErgoScriptCollector()
    return await collector.collect_ergoscript_examples()
