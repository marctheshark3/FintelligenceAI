"""
Data Collection Module

Collects ErgoScript examples and documentation from various sources.
"""

import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin

import aiohttp
import httpx
from pydantic import BaseModel

from ..config import get_settings
from ..rag.models import Document

logger = logging.getLogger(__name__)
settings = get_settings()


class GitHubFile(BaseModel):
    """Represents a file from GitHub repository."""
    name: str
    path: str
    download_url: str
    content: Optional[str] = None
    metadata: Dict = {}


class GitHubDataCollector:
    """Collects data from GitHub repositories."""
    
    def __init__(self):
        self.session: Optional[httpx.AsyncClient] = None
        self.base_url = "https://api.github.com"
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers={"Accept": "application/vnd.github.v3+json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def get_repository_files(
        self,
        owner: str,
        repo: str,
        path: str = "",
        file_extensions: Optional[Set[str]] = None
    ) -> List[GitHubFile]:
        """Get files from a GitHub repository."""
        if not self.session:
            raise RuntimeError("GitHubDataCollector must be used as async context manager")
        
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
            response = await self.session.get(url)
            response.raise_for_status()
            
            contents = response.json()
            files = []
            
            for item in contents:
                if item["type"] == "file":
                    # Filter by file extensions if specified
                    if file_extensions:
                        file_ext = Path(item["name"]).suffix.lower()
                        if file_ext not in file_extensions:
                            continue
                    
                    files.append(GitHubFile(
                        name=item["name"],
                        path=item["path"],
                        download_url=item["download_url"],
                        metadata={
                            "size": item["size"],
                            "sha": item["sha"],
                            "url": item["html_url"]
                        }
                    ))
                elif item["type"] == "dir":
                    # Recursively get files from subdirectories
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
            raise RuntimeError("GitHubDataCollector must be used as async context manager")
        
        try:
            response = await self.session.get(file.download_url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error downloading file {file.name}: {e}")
            return ""


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
            "expert": 4
        }
    
    async def collect_ergoscript_examples(self) -> List[Document]:
        """Collect all ErgoScript examples from the repository."""
        documents = []
        
        async with self.github_collector:
            # Get all markdown files
            files = await self.github_collector.get_repository_files(
                self.repo_owner,
                self.repo_name,
                file_extensions={".md"}
            )
            
            # Filter out README and template files
            example_files = [
                f for f in files 
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
    
    async def _process_ergoscript_example(self, file: GitHubFile, content: str) -> Optional[Document]:
        """Process a single ErgoScript example file."""
        try:
            # Extract metadata from content
            metadata = self._extract_metadata(file, content)
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(content)
            
            # Clean and structure content
            cleaned_content = self._clean_content(content)
            
            # Create DocumentMetadata with proper enum values
            from ..rag.models import DocumentMetadata
            
            # Map difficulty to complexity enum
            complexity_mapping = {
                "starter": "beginner",
                "beginner": "beginner", 
                "intermediate": "intermediate",
                "expert": "advanced"
            }
            
            doc_metadata = DocumentMetadata(
                source="examples",  # Valid enum value
                category="examples",  # Valid enum value - these are ErgoScript examples
                complexity=complexity_mapping.get(metadata["difficulty"].lower(), "beginner"),
                tags=["ergoscript"] + metadata.get("tags", []),
                file_path=file.path,
                url=file.metadata.get("url", ""),
                language="ergoscript",
                tested=False
            )
            
            document = Document(
                id=f"ergoscript_example_{file.name.replace('.md', '')}",
                content=cleaned_content,
                title=metadata["title"],
                metadata=doc_metadata
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
            return None
    
    def _extract_metadata(self, file: GitHubFile, content: str) -> Dict:
        """Extract metadata from the example content."""
        metadata = {
            "title": self._extract_title(content) or file.name.replace('.md', '').replace('_', ' ').title(),
            "difficulty": self._extract_difficulty(content),
            "tags": self._extract_tags(content),
            "use_cases": self._extract_use_cases(content)
        }
        
        # Add difficulty score for sorting
        difficulty_text = metadata["difficulty"].lower()
        metadata["difficulty_score"] = self.difficulty_map.get(difficulty_text, 2)
        
        return metadata
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract title from markdown content."""
        # Look for main heading
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
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
    
    def _extract_tags(self, content: str) -> List[str]:
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
            "multisig": ["multisig", "multi-signature"]
        }
        
        for tag, patterns in tag_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                tags.append(tag)
        
        return tags
    
    def _extract_use_cases(self, content: str) -> List[str]:
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
            "Identity": ["identity", "authentication", "verification"]
        }
        
        for use_case, patterns in use_case_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                use_cases.append(use_case)
        
        return use_cases
    
    def _extract_code_blocks(self, content: str) -> List[Dict]:
        """Extract code blocks from markdown content."""
        code_blocks = []
        
        # Find all code blocks with language specification
        code_pattern = r'```(\w*)\n(.*?)\n```'
        matches = re.findall(code_pattern, content, re.DOTALL)
        
        for i, (language, code) in enumerate(matches):
            code_blocks.append({
                "language": language or "scala",
                "code": code.strip(),
                "block_number": i + 1
            })
        
        return code_blocks
    
    def _clean_content(self, content: str) -> str:
        """Clean and structure the content for better retrieval."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Remove markdown artifacts that might interfere with retrieval
        content = re.sub(r'^\[.*?\]:\s*.*$', '', content, flags=re.MULTILINE)
        
        # Ensure proper spacing around headers
        content = re.sub(r'^(#{1,6})\s*(.+)$', r'\1 \2', content, flags=re.MULTILINE)
        
        return content.strip()


# Convenience function for easy collection
async def collect_ergoscript_knowledge_base() -> List[Document]:
    """Collect the complete ErgoScript knowledge base."""
    collector = ErgoScriptCollector()
    return await collector.collect_ergoscript_examples()