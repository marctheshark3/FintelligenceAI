"""
Document Processing Module

Processes collected documents for optimal retrieval and generation.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from pydantic import BaseModel

from ..rag.models import Document

logger = logging.getLogger(__name__)


class ProcessingStats(BaseModel):
    """Statistics from document processing."""
    total_documents: int
    processed_documents: int
    total_chunks: int
    average_chunk_size: int
    processing_time_seconds: float


class DocumentChunk(BaseModel):
    """Represents a chunk of a document."""
    id: str
    parent_document_id: str
    content: str
    metadata: Dict
    chunk_index: int
    chunk_type: str  # "text", "code", "mixed"


class DocumentProcessor:
    """Base document processor with common functionality."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def process_documents(self, documents: List[Document]) -> Tuple[List[DocumentChunk], ProcessingStats]:
        """Process a list of documents into chunks."""
        start_time = datetime.utcnow()
        all_chunks = []
        processed_count = 0
        
        for document in documents:
            try:
                chunks = self.process_document(document)
                all_chunks.extend(chunks)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing document {document.id}: {e}")
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        stats = ProcessingStats(
            total_documents=len(documents),
            processed_documents=processed_count,
            total_chunks=len(all_chunks),
            average_chunk_size=sum(len(chunk.content) for chunk in all_chunks) // len(all_chunks) if all_chunks else 0,
            processing_time_seconds=processing_time
        )
        
        return all_chunks, stats
    
    def process_document(self, document: Document) -> List[DocumentChunk]:
        """Process a single document into chunks."""
        chunks = []
        
        # Split content into sections
        sections = self._split_into_sections(document.content)
        
        chunk_index = 0
        for section_type, section_content in sections:
            section_chunks = self._chunk_section(
                document, section_content, section_type, chunk_index
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        return chunks
    
    def _split_into_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split content into logical sections."""
        sections = []
        
        # Split by headers first
        header_pattern = r'^(#{1,6})\s+(.+)$'
        parts = re.split(header_pattern, content, flags=re.MULTILINE)
        
        current_section = ""
        current_type = "text"
        
        for i, part in enumerate(parts):
            if re.match(r'^#{1,6}$', part):
                # This is a header level marker
                if current_section.strip():
                    sections.append((current_type, current_section.strip()))
                current_section = parts[i + 1] + "\n" if i + 1 < len(parts) else ""
                current_type = "text"
            elif not re.match(r'^#{1,6}$', part) and not re.match(header_pattern, part):
                current_section += part
        
        # Add the last section
        if current_section.strip():
            sections.append((current_type, current_section.strip()))
        
        return sections if sections else [("text", content)]
    
    def _chunk_section(
        self, 
        document: Document, 
        content: str, 
        section_type: str, 
        start_index: int
    ) -> List[DocumentChunk]:
        """Chunk a section of content."""
        chunks = []
        
        # Simple chunking by character count with overlap
        words = content.split()
        current_chunk = []
        current_size = 0
        chunk_index = start_index
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            
            if current_size + word_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_content = " ".join(current_chunk)
                if len(chunk_content) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        document, chunk_content, section_type, chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-self.chunk_overlap // 10:] if self.chunk_overlap > 0 else []
                current_chunk = overlap_words + [word]
                current_size = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_size += word_size
        
        # Add final chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            if len(chunk_content) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    document, chunk_content, section_type, chunk_index
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self, 
        document: Document, 
        content: str, 
        chunk_type: str, 
        chunk_index: int
    ) -> DocumentChunk:
        """Create a document chunk."""
        chunk_id = f"{document.id}_chunk_{chunk_index}"
        
        chunk_metadata = {
            **document.metadata.dict(),
            "chunk_size": len(content),
            "chunk_word_count": len(content.split()),
            "parent_document_title": getattr(document, "title", ""),
            "parent_document_source": document.metadata.source
        }
        
        return DocumentChunk(
            id=chunk_id,
            parent_document_id=document.id,
            content=content,
            metadata=chunk_metadata,
            chunk_index=chunk_index,
            chunk_type=chunk_type
        )


class ErgoScriptProcessor(DocumentProcessor):
    """Specialized processor for ErgoScript documents."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.code_block_size = 500  # Smaller chunks for code
    
    def _split_into_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split ErgoScript content with special handling for code blocks."""
        sections = []
        
        # First, extract code blocks
        code_pattern = r'```[\w]*\n(.*?)\n```'
        code_blocks = list(re.finditer(code_pattern, content, re.DOTALL))
        
        last_end = 0
        for match in code_blocks:
            # Add text before code block
            text_before = content[last_end:match.start()].strip()
            if text_before:
                sections.append(("text", text_before))
            
            # Add code block
            code_content = match.group(1).strip()
            if code_content:
                sections.append(("code", code_content))
            
            last_end = match.end()
        
        # Add remaining text
        remaining_text = content[last_end:].strip()
        if remaining_text:
            sections.append(("text", remaining_text))
        
        return sections if sections else [("text", content)]
    
    def _chunk_section(
        self, 
        document: Document, 
        content: str, 
        section_type: str, 
        start_index: int
    ) -> List[DocumentChunk]:
        """Chunk ErgoScript sections with type-specific logic."""
        if section_type == "code":
            return self._chunk_code_section(document, content, start_index)
        else:
            return super()._chunk_section(document, content, section_type, start_index)
    
    def _chunk_code_section(
        self, 
        document: Document, 
        content: str, 
        start_index: int
    ) -> List[DocumentChunk]:
        """Chunk code with preservation of logical structures."""
        chunks = []
        
        # Try to split by logical code boundaries
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        chunk_index = start_index
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > self.code_block_size and current_chunk:
                # Create code chunk
                chunk_content = '\n'.join(current_chunk)
                chunk = self._create_code_chunk(document, chunk_content, chunk_index)
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunk = self._create_code_chunk(document, chunk_content, chunk_index)
            chunks.append(chunk)
        
        return chunks
    
    def _create_code_chunk(
        self, 
        document: Document, 
        content: str, 
        chunk_index: int
    ) -> DocumentChunk:
        """Create a code-specific chunk with enhanced metadata."""
        chunk_id = f"{document.id}_code_chunk_{chunk_index}"
        
        # Analyze code content
        code_metadata = self._analyze_code_content(content)
        
        chunk_metadata = {
            **document.metadata.dict(),
            **code_metadata,
            "chunk_size": len(content),
            "chunk_line_count": len(content.split('\n')),
            "parent_document_title": getattr(document, "title", ""),
            "parent_document_source": document.metadata.source
        }
        
        return DocumentChunk(
            id=chunk_id,
            parent_document_id=document.id,
            content=content,
            metadata=chunk_metadata,
            chunk_index=chunk_index,
            chunk_type="code"
        )
    
    def _analyze_code_content(self, code: str) -> Dict:
        """Analyze code content for enhanced metadata."""
        metadata = {}
        
        # Count various ErgoScript constructs
        metadata["has_function_def"] = "def " in code
        metadata["has_contract"] = any(keyword in code for keyword in ["sigmaProp", "proveDlog", "SELF"])
        metadata["has_conditional"] = any(keyword in code for keyword in ["if", "else", "match"])
        metadata["has_crypto"] = any(keyword in code for keyword in ["hash", "signature", "proveDlog"])
        metadata["has_tokens"] = any(keyword in code for keyword in ["TOKENS", "token", "asset"])
        metadata["complexity_score"] = self._calculate_complexity_score(code)
        
        return metadata
    
    def _calculate_complexity_score(self, code: str) -> int:
        """Calculate a simple complexity score for code."""
        score = 0
        
        # Add points for various constructs
        score += code.count("if") * 2
        score += code.count("match") * 3
        score += code.count("def") * 2
        score += code.count("sigmaProp") * 1
        score += code.count("proveDlog") * 2
        score += len(re.findall(r'\{.*?\}', code)) * 1  # Blocks
        
        return min(score, 20)  # Cap at 20