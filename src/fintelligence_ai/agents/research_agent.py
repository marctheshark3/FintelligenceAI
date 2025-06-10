"""
Research Agent for FintelligenceAI.

This module implements the ResearchAgent that handles research queries,
documentation lookup, and information gathering tasks using the RAG pipeline.
"""

import asyncio
from typing import Any, Dict, List, Optional

import dspy

from fintelligence_ai.rag.pipeline import RAGPipeline
from fintelligence_ai.rag.retrieval import RetrievalEngine

from .base import BaseAgent
from .types import (
    AgentCapability,
    AgentConfig,
    AgentRole,
    ConversationContext,
    TaskType,
)


class DocumentationSearch(dspy.Signature):
    """Search for relevant documentation based on a query."""
    
    query: str = dspy.InputField(desc="Search query for documentation")
    context: str = dspy.InputField(desc="Additional context for the search")
    
    relevant_docs: str = dspy.OutputField(desc="Relevant documentation found")
    summary: str = dspy.OutputField(desc="Summary of the findings")
    confidence: float = dspy.OutputField(desc="Confidence score (0-1)")


class ExampleSearch(dspy.Signature):
    """Search for relevant code examples."""
    
    query: str = dspy.InputField(desc="Query describing the needed example")
    use_case: str = dspy.InputField(desc="Specific use case or domain")
    
    examples: str = dspy.OutputField(desc="Relevant code examples")
    explanation: str = dspy.OutputField(desc="Explanation of the examples")
    complexity: str = dspy.OutputField(desc="Complexity level of examples")


class ResearchQuery(dspy.Signature):
    """Perform comprehensive research on a topic."""
    
    topic: str = dspy.InputField(desc="Research topic or question")
    scope: str = dspy.InputField(desc="Scope and depth of research required")
    context: str = dspy.InputField(desc="Additional context for research")
    
    findings: str = dspy.OutputField(desc="Research findings and analysis")
    sources: str = dspy.OutputField(desc="Sources and references")
    recommendations: str = dspy.OutputField(desc="Recommendations based on research")


class ResearchAgent(BaseAgent):
    """
    Research Agent specialized in information gathering and documentation lookup.
    
    This agent handles research queries, documentation searches, and example
    retrieval using the RAG pipeline and vector search capabilities.
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        retrieval_engine: Optional[RetrievalEngine] = None
    ):
        """
        Initialize the Research Agent.
        
        Args:
            config: Agent configuration, will create default if None
            rag_pipeline: RAG pipeline for retrieval and generation
            retrieval_engine: Retrieval engine for document search
        """
        if config is None:
            config = self._create_default_config()
        
        super().__init__(config)
        
        # Initialize RAG components
        self.rag_pipeline = rag_pipeline
        self.retrieval_engine = retrieval_engine
        
        # Initialize DSPy modules
        self._initialize_dspy_modules()
        
        self.logger.info("Research Agent initialized with RAG capabilities")
    
    def _create_default_config(self) -> AgentConfig:
        """Create default configuration for Research Agent."""
        capabilities = [
            AgentCapability(
                name="documentation_search",
                description="Search through documentation and knowledge base",
                supported_task_types=[
                    TaskType.DOCUMENTATION_LOOKUP,
                    TaskType.RESEARCH_QUERY
                ],
                required_tools=["vector_search", "rag_pipeline"]
            ),
            AgentCapability(
                name="example_search",
                description="Find relevant code examples and patterns",
                supported_task_types=[
                    TaskType.EXAMPLE_SEARCH,
                    TaskType.RESEARCH_QUERY
                ],
                required_tools=["vector_search", "example_database"]
            ),
            AgentCapability(
                name="comprehensive_research",
                description="Perform comprehensive research on topics",
                supported_task_types=[TaskType.RESEARCH_QUERY],
                required_tools=["rag_pipeline", "vector_search", "web_search"]
            )
        ]
        
        return AgentConfig(
            agent_id="research_agent_001",
            role=AgentRole.RESEARCH,
            name="Research Agent",
            description="Specialized agent for research, documentation lookup, and information gathering",
            capabilities=capabilities,
            max_concurrent_tasks=3,
            timeout_seconds=180,
            llm_config={
                "temperature": 0.1,
                "max_tokens": 2048
            }
        )
    
    def _initialize_dspy_modules(self) -> None:
        """Initialize DSPy modules for research tasks."""
        self.dspy_modules.update({
            "doc_search": dspy.ChainOfThought(DocumentationSearch),
            "example_search": dspy.ChainOfThought(ExampleSearch),
            "research_query": dspy.ChainOfThought(ResearchQuery),
            "simple_search": dspy.Retrieve(k=5),  # Simple retrieval module
        })
    
    async def _execute_task_impl(
        self,
        task_type: TaskType,
        content: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute research-specific tasks.
        
        Args:
            task_type: Type of research task
            content: Task content/query
            context: Conversation context
            metadata: Additional metadata
            
        Returns:
            Task result data
        """
        if task_type == TaskType.DOCUMENTATION_LOOKUP:
            return await self._handle_documentation_lookup(content, context, metadata)
        elif task_type == TaskType.EXAMPLE_SEARCH:
            return await self._handle_example_search(content, context, metadata)
        elif task_type == TaskType.RESEARCH_QUERY:
            return await self._handle_research_query(content, context, metadata)
        else:
            raise ValueError(f"Unsupported task type for Research Agent: {task_type}")
    
    async def _handle_documentation_lookup(
        self,
        query: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle documentation lookup requests.
        
        Args:
            query: Documentation search query
            context: Conversation context
            metadata: Additional metadata
            
        Returns:
            Documentation lookup results
        """
        self.logger.info(f"Performing documentation lookup for: {query}")
        
        try:
            # Extract context information
            additional_context = ""
            if context and context.context_data:
                additional_context = context.context_data.get("domain", "")
            
            # Use retrieval engine if available
            retrieved_docs = []
            if self.retrieval_engine:
                retrieved_docs = await self._perform_vector_search(query, k=5)
            
            # Use DSPy module for structured search
            doc_search = self.dspy_modules["doc_search"]
            result = doc_search(
                query=query,
                context=additional_context
            )
            
            return {
                "query": query,
                "relevant_docs": result.relevant_docs,
                "summary": result.summary,
                "confidence": result.confidence,
                "retrieved_docs": retrieved_docs,
                "metadata": {
                    "search_type": "documentation",
                    "timestamp": metadata.get("timestamp") if metadata else None,
                    "context_used": bool(additional_context)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in documentation lookup: {e}")
            raise
    
    async def _handle_example_search(
        self,
        query: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle example search requests.
        
        Args:
            query: Example search query
            context: Conversation context
            metadata: Additional metadata
            
        Returns:
            Example search results
        """
        self.logger.info(f"Searching for examples: {query}")
        
        try:
            # Determine use case from context
            use_case = "general"
            if context and context.context_data:
                use_case = context.context_data.get("use_case", "general")
            elif metadata:
                use_case = metadata.get("use_case", "general")
            
            # Search for examples using vector retrieval
            example_docs = []
            if self.vector_retriever:
                example_docs = await self._perform_vector_search(
                    query,
                    k=3,
                    filter_metadata={"type": "example"}
                )
            
            # Use DSPy module for example search
            example_search = self.dspy_modules["example_search"]
            result = example_search(
                query=query,
                use_case=use_case
            )
            
            return {
                "query": query,
                "use_case": use_case,
                "examples": result.examples,
                "explanation": result.explanation,
                "complexity": result.complexity,
                "retrieved_examples": example_docs,
                "metadata": {
                    "search_type": "examples",
                    "use_case": use_case,
                    "example_count": len(example_docs)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in example search: {e}")
            raise
    
    async def _handle_research_query(
        self,
        topic: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle comprehensive research queries.
        
        Args:
            topic: Research topic
            context: Conversation context
            metadata: Additional metadata
            
        Returns:
            Research results
        """
        self.logger.info(f"Performing research on: {topic}")
        
        try:
            # Determine research scope
            scope = metadata.get("scope", "comprehensive") if metadata else "comprehensive"
            additional_context = ""
            
            if context and context.context_data:
                additional_context = str(context.context_data)
            
            # Perform multi-source research
            research_tasks = [
                self._research_documentation(topic),
                self._research_examples(topic),
            ]
            
            # If RAG pipeline is available, use it for enhanced research
            if self.rag_pipeline:
                research_tasks.append(self._research_with_rag(topic, scope))
            
            research_results = await asyncio.gather(*research_tasks, return_exceptions=True)
            
            # Use DSPy module for final synthesis
            research_query = self.dspy_modules["research_query"]
            synthesis = research_query(
                topic=topic,
                scope=scope,
                context=additional_context
            )
            
            return {
                "topic": topic,
                "scope": scope,
                "findings": synthesis.findings,
                "sources": synthesis.sources,
                "recommendations": synthesis.recommendations,
                "raw_research": {
                    "documentation": research_results[0] if len(research_results) > 0 else None,
                    "examples": research_results[1] if len(research_results) > 1 else None,
                    "rag_results": research_results[2] if len(research_results) > 2 else None
                },
                "metadata": {
                    "research_type": "comprehensive",
                    "scope": scope,
                    "sources_count": len([r for r in research_results if not isinstance(r, Exception)])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in research query: {e}")
            raise
    
    async def _perform_vector_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search using the retriever.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of retrieved documents
        """
        if not self.vector_retriever:
            return []
        
        try:
            results = await self.vector_retriever.retrieve(
                query=query,
                k=k,
                filter_metadata=filter_metadata
            )
            return results
        except Exception as e:
            self.logger.error(f"Vector search error: {e}")
            return []
    
    async def _research_documentation(self, topic: str) -> Dict[str, Any]:
        """Research documentation sources."""
        try:
            docs = await self._perform_vector_search(
                topic,
                k=3,
                filter_metadata={"source": "documentation"}
            )
            return {"type": "documentation", "results": docs}
        except Exception as e:
            return {"type": "documentation", "error": str(e)}
    
    async def _research_examples(self, topic: str) -> Dict[str, Any]:
        """Research example sources."""
        try:
            examples = await self._perform_vector_search(
                topic,
                k=3,
                filter_metadata={"type": "example"}
            )
            return {"type": "examples", "results": examples}
        except Exception as e:
            return {"type": "examples", "error": str(e)}
    
    async def _research_with_rag(self, topic: str, scope: str) -> Dict[str, Any]:
        """Research using RAG pipeline."""
        try:
            if not self.rag_pipeline:
                return {"type": "rag", "error": "RAG pipeline not available"}
            
            result = await self.rag_pipeline.process_query(
                query=topic,
                context={"scope": scope}
            )
            return {"type": "rag", "results": result}
        except Exception as e:
            return {"type": "rag", "error": str(e)} 