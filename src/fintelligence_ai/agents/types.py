"""
Type definitions for the AI Agent Framework.

This module defines the core types, enums, and data structures used
throughout the agent system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Enumeration of agent roles in the system."""
    
    ORCHESTRATOR = "orchestrator"
    RESEARCH = "research" 
    GENERATION = "generation"
    VALIDATION = "validation"


class TaskType(str, Enum):
    """Enumeration of task types that agents can handle."""
    
    RESEARCH_QUERY = "research_query"
    CODE_GENERATION = "code_generation"
    CODE_VALIDATION = "code_validation"
    DOCUMENTATION_LOOKUP = "documentation_lookup"
    EXAMPLE_SEARCH = "example_search"
    SYNTAX_CHECK = "syntax_check"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    OPTIMIZATION_SUGGESTION = "optimization_suggestion"


class AgentStatus(str, Enum):
    """Enumeration of agent status states."""
    
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


class MessageType(str, Enum):
    """Enumeration of message types between agents."""
    
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class AgentMessage(BaseModel):
    """Message structure for inter-agent communication."""
    
    id: UUID = Field(default_factory=uuid4, description="Unique message identifier")
    message_type: MessageType = Field(description="Type of message")
    sender_id: str = Field(description="ID of the sending agent")
    recipient_id: Optional[str] = Field(default=None, description="ID of the receiving agent")
    task_type: Optional[TaskType] = Field(default=None, description="Type of task being requested")
    content: str = Field(description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    correlation_id: Optional[UUID] = Field(default=None, description="ID for tracking related messages")


class TaskResult(BaseModel):
    """Result structure for completed agent tasks."""
    
    task_id: UUID = Field(description="Unique task identifier")
    task_type: TaskType = Field(description="Type of task completed")
    agent_id: str = Field(description="ID of the agent that completed the task")
    success: bool = Field(description="Whether the task completed successfully")
    result: Any = Field(description="Task result data")
    error_message: Optional[str] = Field(default=None, description="Error message if task failed")
    execution_time_seconds: float = Field(description="Task execution time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Task completion timestamp")


class AgentCapability(BaseModel):
    """Definition of an agent's capability."""
    
    name: str = Field(description="Capability name")
    description: str = Field(description="Capability description")
    supported_task_types: List[TaskType] = Field(description="Task types this capability supports")
    required_tools: List[str] = Field(default_factory=list, description="Required tools for this capability")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Capability parameters")


class AgentConfig(BaseModel):
    """Configuration for an agent instance."""
    
    agent_id: str = Field(description="Unique agent identifier")
    role: AgentRole = Field(description="Agent role")
    name: str = Field(description="Human-readable agent name")
    description: str = Field(description="Agent description")
    capabilities: List[AgentCapability] = Field(description="Agent capabilities")
    max_concurrent_tasks: int = Field(default=5, description="Maximum concurrent tasks")
    timeout_seconds: int = Field(default=300, description="Task timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed tasks")
    llm_config: Dict[str, Any] = Field(default_factory=dict, description="LLM model configuration")
    tools_config: Dict[str, Any] = Field(default_factory=dict, description="Tools configuration")


class ConversationContext(BaseModel):
    """Context for maintaining conversation state across agent interactions."""
    
    session_id: UUID = Field(description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    conversation_history: List[AgentMessage] = Field(default_factory=list, description="Message history")
    current_task: Optional[TaskType] = Field(default=None, description="Current task being processed")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Contextual data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Context creation time")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


class ValidationCriteria(BaseModel):
    """Criteria for validating generated ErgoScript code."""
    
    syntax_check: bool = Field(default=True, description="Check syntax validity")
    semantic_check: bool = Field(default=True, description="Check semantic correctness") 
    security_check: bool = Field(default=True, description="Check for security issues")
    optimization_check: bool = Field(default=False, description="Check for optimization opportunities")
    gas_estimation: bool = Field(default=True, description="Estimate gas costs")
    compile_test: bool = Field(default=True, description="Attempt compilation")
    test_scenarios: List[str] = Field(default_factory=list, description="Test scenarios to validate")


class ErgoScriptRequest(BaseModel):
    """Request structure for ErgoScript generation."""
    
    description: str = Field(description="Natural language description of the required script")
    use_case: Optional[str] = Field(default=None, description="Specific use case (token, auction, etc.)")
    complexity_level: str = Field(default="intermediate", description="Complexity level: beginner, intermediate, advanced")
    requirements: List[str] = Field(default_factory=list, description="Specific requirements")
    constraints: List[str] = Field(default_factory=list, description="Constraints to consider")
    validation_criteria: ValidationCriteria = Field(default_factory=ValidationCriteria, description="Validation criteria")
    context: Optional[str] = Field(default=None, description="Additional context")


class ErgoScriptResponse(BaseModel):
    """Response structure for ErgoScript generation."""
    
    generated_code: str = Field(description="Generated ErgoScript code")
    explanation: str = Field(description="Explanation of the generated code")
    complexity_score: float = Field(description="Complexity score (1-10)")
    validation_results: Dict[str, Any] = Field(description="Validation results")
    optimization_suggestions: List[str] = Field(default_factory=list, description="Optimization suggestions")
    gas_estimate: Optional[float] = Field(default=None, description="Estimated gas cost")
    documentation_links: List[str] = Field(default_factory=list, description="Relevant documentation links")
    example_usage: Optional[str] = Field(default=None, description="Example usage of the script")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata") 