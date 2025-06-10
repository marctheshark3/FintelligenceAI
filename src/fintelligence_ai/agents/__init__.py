"""
AI Agent Framework for FintelligenceAI.

This module provides a comprehensive multi-agent system for ErgoScript generation
with specialized agents for research, generation, and validation tasks.
"""

from .base import BaseAgent
from .generation_agent import GenerationAgent
from .orchestrator import AgentOrchestrator
from .research_agent import ResearchAgent
from .types import (
    AgentMessage,
    AgentRole,
    ConversationContext,
    ErgoScriptRequest,
    ErgoScriptResponse,
    TaskType,
)
from .validation_agent import ValidationAgent

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
