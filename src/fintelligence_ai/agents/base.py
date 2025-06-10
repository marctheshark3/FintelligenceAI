"""
Base Agent class for the FintelligenceAI agent framework.

This module provides the foundational BaseAgent class that all specialized
agents inherit from, providing common functionality and interfaces.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import dspy
from pydantic import BaseModel

from .types import (
    AgentCapability,
    AgentConfig,
    AgentMessage,
    AgentRole,
    AgentStatus,
    ConversationContext,
    MessageType,
    TaskResult,
    TaskType,
)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the FintelligenceAI system.
    
    Provides common functionality for task execution, communication,
    and integration with the DSPy framework.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the base agent.
        
        Args:
            config: Agent configuration containing role, capabilities, etc.
        """
        self.config = config
        self.agent_id = config.agent_id
        self.role = config.role
        self.name = config.name
        self.status = AgentStatus.IDLE
        
        # Initialize logging
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        
        # Task management
        self.active_tasks: Dict[UUID, asyncio.Task] = {}
        self.completed_tasks: List[TaskResult] = []
        
        # Communication
        self.message_handlers: Dict[MessageType, callable] = {
            MessageType.REQUEST: self._handle_request,
            MessageType.RESPONSE: self._handle_response,
            MessageType.NOTIFICATION: self._handle_notification,
            MessageType.ERROR: self._handle_error,
        }
        
        # DSPy modules (to be initialized by subclasses)
        self.dspy_modules: Dict[str, dspy.Module] = {}
        
        self.logger.info(f"Initialized {self.role.value} agent: {self.name}")
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities."""
        return self.config.capabilities
    
    @property
    def supported_task_types(self) -> List[TaskType]:
        """Get list of supported task types."""
        task_types = []
        for capability in self.capabilities:
            task_types.extend(capability.supported_task_types)
        return list(set(task_types))
    
    def can_handle_task(self, task_type: TaskType) -> bool:
        """
        Check if agent can handle a specific task type.
        
        Args:
            task_type: Type of task to check
            
        Returns:
            True if agent can handle the task type
        """
        return task_type in self.supported_task_types
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message.
        
        Args:
            message: Incoming message to process
            
        Returns:
            Optional response message
        """
        self.logger.debug(f"Processing message: {message.message_type} from {message.sender_id}")
        
        try:
            handler = self.message_handlers.get(message.message_type)
            if handler:
                return await handler(message)
            else:
                self.logger.warning(f"No handler for message type: {message.message_type}")
                return None
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return self._create_error_message(
                original_message=message,
                error_message=str(e)
            )
    
    async def execute_task(
        self,
        task_type: TaskType,
        content: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """
        Execute a task asynchronously.
        
        Args:
            task_type: Type of task to execute
            content: Task content/prompt
            context: Optional conversation context
            metadata: Optional task metadata
            
        Returns:
            Task execution result
        """
        task_id = uuid4()
        start_time = time.time()
        
        self.logger.info(f"Starting task {task_id}: {task_type}")
        self.status = AgentStatus.PROCESSING
        
        try:
            if not self.can_handle_task(task_type):
                raise ValueError(f"Agent {self.agent_id} cannot handle task type: {task_type}")
            
            # Execute the specific task implementation
            result = await self._execute_task_impl(task_type, content, context, metadata)
            
            execution_time = time.time() - start_time
            
            task_result = TaskResult(
                task_id=task_id,
                task_type=task_type,
                agent_id=self.agent_id,
                success=True,
                result=result,
                execution_time_seconds=execution_time,
                metadata=metadata or {}
            )
            
            self.completed_tasks.append(task_result)
            self.logger.info(f"Completed task {task_id} in {execution_time:.2f}s")
            
            return task_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            self.logger.error(f"Task {task_id} failed: {error_message}")
            
            task_result = TaskResult(
                task_id=task_id,
                task_type=task_type,
                agent_id=self.agent_id,
                success=False,
                result=None,
                error_message=error_message,
                execution_time_seconds=execution_time,
                metadata=metadata or {}
            )
            
            self.completed_tasks.append(task_result)
            return task_result
            
        finally:
            self.status = AgentStatus.IDLE
    
    @abstractmethod
    async def _execute_task_impl(
        self,
        task_type: TaskType,
        content: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Abstract method for implementing task execution logic.
        
        Must be implemented by subclasses to handle specific task types.
        
        Args:
            task_type: Type of task to execute
            content: Task content/prompt
            context: Optional conversation context
            metadata: Optional task metadata
            
        Returns:
            Task result data
        """
        pass
    
    async def _handle_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming request messages."""
        if message.task_type:
            # Execute the requested task
            result = await self.execute_task(
                task_type=message.task_type,
                content=message.content,
                metadata=message.metadata
            )
            
            # Create response message
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                content=str(result.result) if result.success else f"Error: {result.error_message}",
                metadata={
                    "task_result": result.dict(),
                    "correlation_id": str(message.correlation_id) if message.correlation_id else None
                },
                correlation_id=message.correlation_id
            )
        else:
            self.logger.warning("Received request message without task_type")
            return None
    
    async def _handle_response(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming response messages."""
        self.logger.debug(f"Received response from {message.sender_id}")
        # Base implementation just logs; subclasses can override for specific handling
        return None
    
    async def _handle_notification(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming notification messages."""
        self.logger.info(f"Received notification from {message.sender_id}: {message.content}")
        return None
    
    async def _handle_error(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming error messages."""
        self.logger.error(f"Received error from {message.sender_id}: {message.content}")
        return None
    
    def _create_error_message(
        self,
        original_message: AgentMessage,
        error_message: str
    ) -> AgentMessage:
        """Create an error response message."""
        return AgentMessage(
            message_type=MessageType.ERROR,
            sender_id=self.agent_id,
            recipient_id=original_message.sender_id,
            content=error_message,
            metadata={"original_message_id": str(original_message.id)},
            correlation_id=original_message.correlation_id
        )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status and statistics.
        
        Returns:
            Dictionary containing agent status information
        """
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "name": self.name,
            "status": self.status.value,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "supported_task_types": [t.value for t in self.supported_task_types],
            "capabilities": [cap.dict() for cap in self.capabilities]
        }
    
    def reset(self) -> None:
        """Reset agent state."""
        self.active_tasks.clear()
        self.completed_tasks.clear()
        self.status = AgentStatus.IDLE
        self.logger.info(f"Reset agent {self.agent_id}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent."""
        self.logger.info(f"Shutting down agent {self.agent_id}")
        
        # Cancel active tasks
        for task in self.active_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        self.status = AgentStatus.IDLE
        self.logger.info(f"Agent {self.agent_id} shutdown complete") 