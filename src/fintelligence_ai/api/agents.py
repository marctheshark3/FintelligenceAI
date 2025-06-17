"""
Agent API endpoints for FintelligenceAI.

This module provides REST API endpoints for interacting with the AI agent system,
including ErgoScript generation, research queries, and code validation.
"""

import logging
from typing import Any, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from fintelligence_ai.agents import (
    AgentOrchestrator,
    ConversationContext,
    ErgoScriptResponse,
    GenerationAgent,
    ResearchAgent,
    TaskType,
    ValidationAgent,
)
from fintelligence_ai.config import get_settings

# Create router
router = APIRouter(prefix="/agents", tags=["AI Agents"])

# Global agent instances (would be managed by dependency injection in production)
_orchestrator: Optional[AgentOrchestrator] = None
_generation_agent: Optional[GenerationAgent] = None
_research_agent: Optional[ResearchAgent] = None
_validation_agent: Optional[ValidationAgent] = None

logger = logging.getLogger(__name__)


# Request/Response Models
class GenerateCodeRequest(BaseModel):
    """Request model for code generation."""

    description: str = Field(
        description="Natural language description of the required code"
    )
    use_case: Optional[str] = Field(
        default=None, description="Specific use case or application type"
    )
    complexity_level: str = Field(
        default="intermediate",
        description="Complexity level: beginner, intermediate, advanced",
    )
    requirements: list[str] = Field(
        default_factory=list, description="Specific requirements"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Constraints to consider"
    )
    session_id: Optional[UUID] = Field(
        default=None, description="Session ID for context"
    )


class ResearchRequest(BaseModel):
    """Request model for research queries."""

    query: str = Field(description="Research query or topic")
    scope: str = Field(default="comprehensive", description="Research scope")
    include_examples: bool = Field(
        default=True, description="Include code examples in research"
    )
    session_id: Optional[UUID] = Field(
        default=None, description="Session ID for context"
    )


class ResearchSummaryRequest(BaseModel):
    """Request model for research summary generation."""

    query: str = Field(description="Summary query or focus area")
    research_context: dict = Field(description="Research context to summarize")
    scope: str = Field(default="focused_summary", description="Summary scope")
    include_examples: bool = Field(
        default=True, description="Include code examples in summary"
    )
    session_id: Optional[UUID] = Field(
        default=None, description="Session ID for context"
    )


class ValidateCodeRequest(BaseModel):
    """Request model for code validation."""

    code: str = Field(description="Code to validate")
    use_case: Optional[str] = Field(default=None, description="Use case context")
    validation_criteria: dict[str, Any] = Field(
        default_factory=lambda: {
            "syntax_check": True,
            "semantic_check": True,
            "security_check": True,
        },
        description="Validation criteria",
    )
    session_id: Optional[UUID] = Field(
        default=None, description="Session ID for context"
    )


class AgentStatusResponse(BaseModel):
    """Response model for agent status."""

    agent_id: str
    role: str
    name: str
    status: str
    active_tasks: int
    completed_tasks: int
    supported_task_types: list[str]


class OrchestrationStatusResponse(BaseModel):
    """Response model for orchestration status."""

    orchestrator: AgentStatusResponse
    agents: dict[str, AgentStatusResponse]


# Dependency to get orchestrator instance
async def get_orchestrator() -> AgentOrchestrator:
    """Get or create the agent orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        logger.info("Initializing Agent Orchestrator")
        _orchestrator = AgentOrchestrator()
    return _orchestrator


async def get_generation_agent() -> GenerationAgent:
    """Get or create the generation agent instance."""
    global _generation_agent
    if _generation_agent is None:
        logger.info("Initializing Generation Agent")

        # Initialize generation agent with default configuration
        # Note: Generation agent uses DSPy modules for code generation
        # RAG integration happens at the orchestrator level
        _generation_agent = GenerationAgent()
        logger.info("Generation Agent initialized successfully")
    return _generation_agent


async def get_research_agent() -> ResearchAgent:
    """Get or create the research agent instance."""
    global _research_agent
    if _research_agent is None:
        logger.info("Initializing Research Agent with RAG pipeline")

        # Import RAG pipeline factory
        from fintelligence_ai.config import get_settings
        from fintelligence_ai.rag.factory import create_default_rag_pipeline

        # Get settings to use the provider-specific collection name
        settings = get_settings()
        collection_name = settings.get_provider_collection_name()
        persist_directory = settings.get_provider_persist_directory()

        # Create RAG pipeline with the provider-specific collection
        rag_pipeline = create_default_rag_pipeline(
            collection_name=collection_name, persist_directory=persist_directory
        )

        # Initialize research agent with RAG pipeline
        _research_agent = ResearchAgent(rag_pipeline=rag_pipeline)
        logger.info(f"Research Agent initialized with collection: {collection_name}")
    return _research_agent


async def get_validation_agent() -> ValidationAgent:
    """Get or create the validation agent instance."""
    global _validation_agent
    if _validation_agent is None:
        logger.info("Initializing Validation Agent")
        _validation_agent = ValidationAgent()
    return _validation_agent


# API Endpoints
@router.post("/generate-code", response_model=ErgoScriptResponse)
async def generate_code(
    request: GenerateCodeRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> ErgoScriptResponse:
    """
    Generate code using the orchestrated agent workflow.

    This endpoint orchestrates research, generation, and validation phases
    to provide comprehensive code generation with context. The system
    automatically adapts to the type of documentation in the knowledge base.
    """
    try:
        logger.info(f"Generating code for: {request.description}")

        # Create conversation context
        context = ConversationContext(
            session_id=request.session_id or uuid4(),
            context_data={
                "use_case": request.use_case,
                "complexity_level": request.complexity_level,
            },
        )

        # Prepare metadata
        metadata = {
            "use_case": request.use_case,
            "complexity_level": request.complexity_level,
            "requirements": request.requirements,
            "constraints": request.constraints,
        }

        # Execute orchestrated workflow
        result = await orchestrator.execute_task(
            task_type=TaskType.CODE_GENERATION,
            content=request.description,
            context=context,
            metadata=metadata,
        )

        if result.success:
            return result.result
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Code generation failed: {result.error_message}",
            )

    except Exception as e:
        logger.error(f"Error generating code: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.post("/generate-code/simple")
async def generate_code_simple(
    request: GenerateCodeRequest,
    generation_agent: GenerationAgent = Depends(get_generation_agent),
) -> ErgoScriptResponse:
    """
    Generate code using only the generation agent (faster, less comprehensive).

    This endpoint bypasses orchestration for faster code generation
    when research and validation are not required. The system automatically
    adapts to the type of documentation in the knowledge base.
    """
    try:
        logger.info(f"Simple code generation for: {request.description}")

        # Create basic context
        context = ConversationContext(
            session_id=request.session_id or uuid4(),
            context_data={"use_case": request.use_case},
        )

        # Prepare metadata
        metadata = {
            "use_case": request.use_case,
            "complexity_level": request.complexity_level,
        }

        # Execute generation task
        result = await generation_agent.execute_task(
            task_type=TaskType.CODE_GENERATION,
            content=request.description,
            context=context,
            metadata=metadata,
        )

        if result.success:
            return result.result
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Code generation failed: {result.error_message}",
            )

    except Exception as e:
        logger.error(f"Error in simple code generation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.post("/research")
async def research_query(
    request: ResearchRequest,
    research_agent: ResearchAgent = Depends(get_research_agent),
) -> dict[str, Any]:
    """
    Perform research query using the research agent.

    This endpoint provides access to documentation search, example lookup,
    and comprehensive research capabilities.
    """
    try:
        logger.info(f"Research query: {request.query}")

        # Create context
        context = ConversationContext(
            session_id=request.session_id or uuid4(),
            context_data={"scope": request.scope},
        )

        # Prepare metadata
        metadata = {
            "scope": request.scope,
            "include_examples": request.include_examples,
        }

        # Execute research task
        result = await research_agent.execute_task(
            task_type=TaskType.RESEARCH_QUERY,
            content=request.query,
            context=context,
            metadata=metadata,
        )

        if result.success:
            return result.result
        else:
            raise HTTPException(
                status_code=500, detail=f"Research failed: {result.error_message}"
            )

    except Exception as e:
        logger.error(f"Error in research query: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.post("/research-summary")
async def research_summary(
    request: ResearchSummaryRequest,
    research_agent: ResearchAgent = Depends(get_research_agent),
) -> dict[str, Any]:
    """
    Generate a focused summary from research context for code generation.

    This endpoint processes research findings into a concise, actionable summary
    that can be used to inform code generation with specific patterns and examples.
    """
    try:
        logger.info(f"Research summary for: {request.query}")

        # Extract research context
        research_context = request.research_context
        findings = research_context.get("findings", "")
        sources = research_context.get("sources", [])
        recommendations = research_context.get("recommendations", "")

        # Create a focused summary query
        summary_query = f"""
        Create a focused summary for ErgoScript code generation based on this research:

        ORIGINAL QUERY: {request.query}

        RESEARCH FINDINGS:
        {findings}

        SOURCES: {', '.join(sources) if isinstance(sources, list) else str(sources)}

        RECOMMENDATIONS:
        {recommendations}

        TASK: Create a concise summary that includes:
        1. Key concepts and patterns relevant to code generation
        2. Specific ErgoScript syntax and methods to use
        3. Best practices and common pitfalls to avoid
        4. Concrete examples if available

        Focus on actionable information that a code generation agent can use directly.
        """

        # Create context for summary generation
        context = ConversationContext(
            session_id=request.session_id or uuid4(),
            context_data={
                "scope": "focused_summary",
                "original_query": request.query,
                "research_context": research_context,
            },
        )

        # Prepare metadata for summary
        metadata = {
            "scope": "focused_summary",
            "include_examples": request.include_examples,
            "original_scope": research_context.get("scope", "comprehensive"),
            "summary_purpose": "code_generation",
        }

        # Execute summary task using research agent
        result = await research_agent.execute_task(
            task_type=TaskType.RESEARCH_QUERY,
            content=summary_query,
            context=context,
            metadata=metadata,
        )

        if result.success:
            # Enhance the result with summary-specific metadata
            summary_result = result.result
            summary_result.update(
                {
                    "summary_type": "code_generation_focused",
                    "original_query": request.query,
                    "original_research_scope": research_context.get(
                        "scope", "comprehensive"
                    ),
                    "summary_created_at": metadata.get("timestamp"),
                    "ready_for_code_generation": True,
                }
            )
            return summary_result
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Summary generation failed: {result.error_message}",
            )

    except Exception as e:
        logger.error(f"Error in research summary: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.post("/validate-code")
async def validate_code(
    request: ValidateCodeRequest,
    validation_agent: ValidationAgent = Depends(get_validation_agent),
) -> dict[str, Any]:
    """
    Validate code using the validation agent.

    This endpoint provides syntax checking, semantic analysis,
    and security assessment for code. The system automatically
    adapts to the type of code being validated.
    """
    try:
        logger.info("Validating code")

        # Create context
        context = ConversationContext(
            session_id=request.session_id or uuid4(),
            context_data={"use_case": request.use_case},
        )

        # Prepare metadata
        metadata = {
            "use_case": request.use_case,
            "validation_criteria": request.validation_criteria,
        }

        # Execute validation task
        result = await validation_agent.execute_task(
            task_type=TaskType.CODE_VALIDATION,
            content=request.code,
            context=context,
            metadata=metadata,
        )

        if result.success:
            return result.result
        else:
            raise HTTPException(
                status_code=500, detail=f"Validation failed: {result.error_message}"
            )

    except Exception as e:
        logger.error(f"Error in code validation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.get("/status", response_model=OrchestrationStatusResponse)
async def get_agent_status(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> OrchestrationStatusResponse:
    """
    Get status of all agents in the system.

    This endpoint provides health monitoring and statistics
    for all agents in the orchestration system.
    """
    try:
        status_data = await orchestrator.get_agent_status()

        # Convert to response model
        orchestrator_status = AgentStatusResponse(**status_data["orchestrator"])
        agent_statuses = {
            name: AgentStatusResponse(**agent_data)
            for name, agent_data in status_data["agents"].items()
        }

        return OrchestrationStatusResponse(
            orchestrator=orchestrator_status, agents=agent_statuses
        )

    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.post("/reset")
async def reset_agents(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> dict[str, str]:
    """
    Reset all agents to initial state.

    This endpoint clears agent state and task history,
    useful for debugging and testing.
    """
    try:
        logger.info("Resetting all agents")

        # Reset orchestrator and all sub-agents
        orchestrator.reset()
        for agent in orchestrator.agents.values():
            agent.reset()

        return {"message": "All agents reset successfully"}

    except Exception as e:
        logger.error(f"Error resetting agents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


# Background task endpoints
@router.post("/generate-code/async")
async def generate_code_async(
    request: GenerateCodeRequest,
    background_tasks: BackgroundTasks,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> dict[str, str]:
    """
    Generate ErgoScript code asynchronously.

    This endpoint starts code generation in the background
    and returns a task ID for later retrieval.
    """
    task_id = str(uuid4())

    async def generate_task():
        """Background task for code generation."""
        try:
            logger.info(f"Background code generation task {task_id} started")

            context = ConversationContext(
                session_id=request.session_id or uuid4(),
                context_data={
                    "use_case": request.use_case,
                    "complexity_level": request.complexity_level,
                },
            )

            metadata = {
                "use_case": request.use_case,
                "complexity_level": request.complexity_level,
                "task_id": task_id,
            }

            await orchestrator.execute_task(
                task_type=TaskType.CODE_GENERATION,
                content=request.description,
                context=context,
                metadata=metadata,
            )

            # In production, would store result in database or cache
            logger.info(f"Background task {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Background task {task_id} failed: {e}")

    background_tasks.add_task(generate_task)

    return {
        "task_id": task_id,
        "message": "Code generation started in background",
        "status": "pending",
    }


# Health check endpoint
@router.get("/health")
async def agent_health_check() -> dict[str, Any]:
    """
    Agent system health check.

    This endpoint provides basic health information
    about the agent system components.
    """
    try:
        settings = get_settings()

        health_status = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            "components": {
                "orchestrator": "available",
                "generation_agent": "available",
                "research_agent": "available",
                "validation_agent": "available",
                "dspy_framework": "available",
            },
            "configuration": {
                "environment": settings.app_environment,
                "debug_mode": settings.app_debug,
            },
        }

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z",
        }
