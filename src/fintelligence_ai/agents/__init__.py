"""
AI Agent Framework for FintelligenceAI.

This module provides a comprehensive multi-agent system for ErgoScript generation
with specialized agents for research, generation, and validation tasks.
"""

from .base import BaseAgent
from .orchestrator import AgentOrchestrator
from .research_agent import ResearchAgent
from .generation_agent import GenerationAgent
from .validation_agent import ValidationAgent
from .types import (
    AgentMessage, 
    AgentRole, 
    TaskType, 
    ErgoScriptRequest, 
    ErgoScriptResponse,
    ConversationContext
)

__all__ = [
    "BaseAgent",
    "AgentOrchestrator", 
    "ResearchAgent",
    "GenerationAgent",
    "ValidationAgent",
    "AgentMessage",
    "AgentRole",
    "TaskType",
    "ErgoScriptRequest",
    "ErgoScriptResponse",
    "ConversationContext",
]
