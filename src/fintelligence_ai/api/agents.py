"""
Agent API endpoints for FintelligenceAI.

This module provides REST API endpoints for interacting with the AI agent system,
including ErgoScript generation, research queries, and code validation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from fintelligence_ai.agents import (
    AgentOrchestrator,
    GenerationAgent,
    ResearchAgent,
    ValidationAgent,
    ErgoScriptRequest,
    ErgoScriptResponse,
    TaskType,
    ConversationContext,
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
    """Request model for ErgoScript code generation."""
    
    description: str = Field(description="Natural language description of the required script")
    use_case: Optional[str] = Field(default=None, description="Specific use case (token, auction, oracle, etc.)")
    complexity_level: str = Field(default="intermediate", description="Complexity level: beginner, intermediate, advanced")
    requirements: List[str] = Field(default_factory=list, description="Specific requirements")
    constraints: List[str] = Field(default_factory=list, description="Constraints to consider")
    session_id: Optional[UUID] = Field(default=None, description="Session ID for context")


class ResearchRequest(BaseModel):
    """Request model for research queries."""
    
    query: str = Field(description="Research query or topic")
    scope: str = Field(default="comprehensive", description="Research scope")
    include_examples: bool = Field(default=True, description="Include code examples in research")
    session_id: Optional[UUID] = Field(default=None, description="Session ID for context")


class ValidateCodeRequest(BaseModel):
    """Request model for code validation."""
    
    code: str = Field(description="ErgoScript code to validate")
    use_case: Optional[str] = Field(default=None, description="Use case context")
    validation_criteria: Dict[str, Any] = Field(
        default_factory=lambda: {
            "syntax_check": True,
            "semantic_check": True,
            "security_check": True
        },
        description="Validation criteria"
    )
    session_id: Optional[UUID] = Field(default=None, description="Session ID for context")


class AgentStatusResponse(BaseModel):
    """Response model for agent status."""
    
    agent_id: str
    role: str
    name: str
    status: str
    active_tasks: int
    completed_tasks: int
    supported_task_types: List[str]


class OrchestrationStatusResponse(BaseModel):
    """Response model for orchestration status."""
    
    orchestrator: AgentStatusResponse
    agents: Dict[str, AgentStatusResponse]


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
        _generation_agent = GenerationAgent()
    return _generation_agent


async def get_research_agent() -> ResearchAgent:
    """Get or create the research agent instance."""
    global _research_agent
    if _research_agent is None:
        logger.info("Initializing Research Agent")
        _research_agent = ResearchAgent()
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
async def generate_ergoscript_code(
    request: GenerateCodeRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> ErgoScriptResponse:
    """
    Generate ErgoScript code using the orchestrated agent workflow.
    
    This endpoint orchestrates research, generation, and validation phases
    to provide comprehensive ErgoScript code generation with context.
    """
    try:
        logger.info(f"Generating ErgoScript code for: {request.description}")
        
        # Create conversation context
        context = ConversationContext(
            session_id=request.session_id or uuid4(),
            context_data={
                "use_case": request.use_case,
                "complexity_level": request.complexity_level
            }
        )
        
        # Prepare metadata
        metadata = {
            "use_case": request.use_case,
            "complexity_level": request.complexity_level,
            "requirements": request.requirements,
            "constraints": request.constraints
        }
        
        # Execute orchestrated workflow
        result = await orchestrator.execute_task(
            task_type=TaskType.CODE_GENERATION,
            content=request.description,
            context=context,
            metadata=metadata
        )
        
        if result.success:
            return result.result
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Code generation failed: {result.error_message}"
            )
            
    except Exception as e:
        logger.error(f"Error generating ErgoScript code: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/generate-code/simple")
async def generate_code_simple(
    request: GenerateCodeRequest,
    generation_agent: GenerationAgent = Depends(get_generation_agent)
) -> ErgoScriptResponse:
    """
    Generate ErgoScript code using only the generation agent (faster, less comprehensive).
    
    This endpoint bypasses orchestration for faster code generation
    when research and validation are not required.
    """
    try:
        logger.info(f"Simple code generation for: {request.description}")
        
        # Create basic context
        context = ConversationContext(
            session_id=request.session_id or uuid4(),
            context_data={"use_case": request.use_case}
        )
        
        # Prepare metadata
        metadata = {
            "use_case": request.use_case,
            "complexity_level": request.complexity_level
        }
        
        # Execute generation task
        result = await generation_agent.execute_task(
            task_type=TaskType.CODE_GENERATION,
            content=request.description,
            context=context,
            metadata=metadata
        )
        
        if result.success:
            return result.result
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Code generation failed: {result.error_message}"
            )
            
    except Exception as e:
        logger.error(f"Error in simple code generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/research")
async def research_query(
    request: ResearchRequest,
    research_agent: ResearchAgent = Depends(get_research_agent)
) -> Dict[str, Any]:
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
            context_data={"scope": request.scope}
        )
        
        # Prepare metadata
        metadata = {
            "scope": request.scope,
            "include_examples": request.include_examples
        }
        
        # Execute research task
        result = await research_agent.execute_task(
            task_type=TaskType.RESEARCH_QUERY,
            content=request.query,
            context=context,
            metadata=metadata
        )
        
        if result.success:
            return result.result
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Research failed: {result.error_message}"
            )
            
    except Exception as e:
        logger.error(f"Error in research query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/validate-code")
async def validate_code(
    request: ValidateCodeRequest,
    validation_agent: ValidationAgent = Depends(get_validation_agent)
) -> Dict[str, Any]:
    """
    Validate ErgoScript code using the validation agent.
    
    This endpoint provides syntax checking, semantic analysis,
    and security assessment for ErgoScript code.
    """
    try:
        logger.info("Validating ErgoScript code")
        
        # Create context
        context = ConversationContext(
            session_id=request.session_id or uuid4(),
            context_data={"use_case": request.use_case}
        )
        
        # Prepare metadata
        metadata = {
            "use_case": request.use_case,
            "validation_criteria": request.validation_criteria
        }
        
        # Execute validation task
        result = await validation_agent.execute_task(
            task_type=TaskType.CODE_VALIDATION,
            content=request.code,
            context=context,
            metadata=metadata
        )
        
        if result.success:
            return result.result
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Validation failed: {result.error_message}"
            )
            
    except Exception as e:
        logger.error(f"Error in code validation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/status", response_model=OrchestrationStatusResponse)
async def get_agent_status(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
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
            orchestrator=orchestrator_status,
            agents=agent_statuses
        )
        
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/reset")
async def reset_agents(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, str]:
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
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# Background task endpoints
@router.post("/generate-code/async")
async def generate_code_async(
    request: GenerateCodeRequest,
    background_tasks: BackgroundTasks,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, str]:
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
                    "complexity_level": request.complexity_level
                }
            )
            
            metadata = {
                "use_case": request.use_case,
                "complexity_level": request.complexity_level,
                "task_id": task_id
            }
            
            result = await orchestrator.execute_task(
                task_type=TaskType.CODE_GENERATION,
                content=request.description,
                context=context,
                metadata=metadata
            )
            
            # In production, would store result in database or cache
            logger.info(f"Background task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Background task {task_id} failed: {e}")
    
    background_tasks.add_task(generate_task)
    
    return {
        "task_id": task_id,
        "message": "Code generation started in background",
        "status": "pending"
    }


# Health check endpoint
@router.get("/health")
async def agent_health_check() -> Dict[str, Any]:
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
                "dspy_framework": "available"
            },
            "configuration": {
                "environment": settings.app_environment,
                "debug_mode": settings.app_debug
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        } 