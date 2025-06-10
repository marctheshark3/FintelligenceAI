"""
Generation Agent for FintelligenceAI.

This module implements the GenerationAgent that specializes in ErgoScript code
generation using DSPy modules and RAG-enhanced context.
"""

import asyncio
from typing import Any, Dict, List, Optional

import dspy

from fintelligence_ai.rag.pipeline import RAGPipeline

from .base import BaseAgent
from .types import (
    AgentCapability,
    AgentConfig,
    AgentRole,
    ConversationContext,
    ErgoScriptRequest,
    ErgoScriptResponse,
    TaskType,
    ValidationCriteria,
)


class ErgoScriptGeneration(dspy.Signature):
    """Generate ErgoScript code based on requirements."""
    
    description: str = dspy.InputField(desc="Natural language description of the required script")
    requirements: str = dspy.InputField(desc="Specific requirements and constraints")
    context: str = dspy.InputField(desc="Additional context and examples")
    use_case: str = dspy.InputField(desc="Specific use case (token, auction, etc.)")
    
    code: str = dspy.OutputField(desc="Generated ErgoScript code")
    explanation: str = dspy.OutputField(desc="Detailed explanation of the code")
    complexity_score: float = dspy.OutputField(desc="Complexity score (1-10)")


class CodeOptimization(dspy.Signature):
    """Optimize existing ErgoScript code."""
    
    original_code: str = dspy.InputField(desc="Original ErgoScript code to optimize")
    optimization_goals: str = dspy.InputField(desc="Optimization objectives")
    constraints: str = dspy.InputField(desc="Constraints to consider")
    
    optimized_code: str = dspy.OutputField(desc="Optimized ErgoScript code")
    improvements: str = dspy.OutputField(desc="List of improvements made")
    gas_savings: float = dspy.OutputField(desc="Estimated gas savings percentage")


class PatternApplication(dspy.Signature):
    """Apply design patterns to ErgoScript code."""
    
    base_code: str = dspy.InputField(desc="Base ErgoScript code")
    patterns: str = dspy.InputField(desc="Design patterns to apply")
    context: str = dspy.InputField(desc="Context for pattern application")
    
    enhanced_code: str = dspy.OutputField(desc="Enhanced code with patterns")
    pattern_explanations: str = dspy.OutputField(desc="Explanation of applied patterns")
    benefits: str = dspy.OutputField(desc="Benefits of the patterns")


class GenerationAgent(BaseAgent):
    """
    Generation Agent specialized in ErgoScript code generation.
    
    This agent handles code generation tasks, optimization suggestions,
    and pattern application using DSPy modules and RAG-enhanced context.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Generation Agent."""
        if config is None:
            config = self._create_default_config()
        
        super().__init__(config)
        self._initialize_dspy_modules()
        self.logger.info("Generation Agent initialized")
    
    def _create_default_config(self) -> AgentConfig:
        """Create default configuration for Generation Agent."""
        capabilities = [
            AgentCapability(
                name="ergoscript_generation",
                description="Generate ErgoScript code from natural language descriptions",
                supported_task_types=[TaskType.CODE_GENERATION],
                required_tools=["dspy_modules"]
            )
        ]
        
        return AgentConfig(
            agent_id="generation_agent_001",
            role=AgentRole.GENERATION,
            name="Generation Agent",
            description="Specialized agent for ErgoScript code generation",
            capabilities=capabilities,
            max_concurrent_tasks=2,
            timeout_seconds=300
        )
    
    def _initialize_dspy_modules(self) -> None:
        """Initialize DSPy modules for code generation tasks."""
        self.dspy_modules.update({
            "code_generator": dspy.ChainOfThought(ErgoScriptGeneration),
        })
    
    async def _execute_task_impl(
        self,
        task_type: TaskType,
        content: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute generation-specific tasks."""
        if task_type == TaskType.CODE_GENERATION:
            return await self._handle_code_generation(content, context, metadata)
        else:
            raise ValueError(f"Unsupported task type for Generation Agent: {task_type}")
    
    async def _handle_code_generation(
        self,
        requirements: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ErgoScriptResponse:
        """Handle ErgoScript code generation requests."""
        self.logger.info("Generating ErgoScript code")
        
        try:
            # Create request
            request = ErgoScriptRequest(
                description=requirements,
                use_case=metadata.get("use_case", "general") if metadata else "general"
            )
            
            # Generate code
            generator = self.dspy_modules["code_generator"]
            result = generator(
                description=request.description,
                requirements=requirements,
                context="",
                use_case=request.use_case
            )
            
            # Create response
            response = ErgoScriptResponse(
                generated_code=result.code,
                explanation=result.explanation,
                complexity_score=result.complexity_score,
                validation_results={}
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in code generation: {e}")
            raise 