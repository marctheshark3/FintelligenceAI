"""
Agent Orchestrator for FintelligenceAI.

This module implements the AgentOrchestrator that coordinates multiple specialized
agents to handle complex tasks like ErgoScript generation with research, generation,
and validation phases.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .base import BaseAgent
from .generation_agent import GenerationAgent
from .research_agent import ResearchAgent
from .validation_agent import ValidationAgent
from .types import (
    AgentCapability,
    AgentConfig,
    AgentMessage,
    AgentRole,
    ConversationContext,
    ErgoScriptRequest,
    ErgoScriptResponse,
    MessageType,
    TaskType,
)


class AgentOrchestrator(BaseAgent):
    """
    Agent Orchestrator that coordinates multiple specialized agents.
    
    This orchestrator manages the workflow of research, generation, and validation
    agents to provide comprehensive ErgoScript generation services.
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        research_agent: Optional[ResearchAgent] = None,
        generation_agent: Optional[GenerationAgent] = None,
        validation_agent: Optional[ValidationAgent] = None
    ):
        """
        Initialize the Agent Orchestrator.
        
        Args:
            config: Orchestrator configuration
            research_agent: Research agent instance
            generation_agent: Generation agent instance  
            validation_agent: Validation agent instance
        """
        if config is None:
            config = self._create_default_config()
        
        super().__init__(config)
        
        # Initialize sub-agents
        self.research_agent = research_agent or ResearchAgent()
        self.generation_agent = generation_agent or GenerationAgent()
        self.validation_agent = validation_agent or ValidationAgent()
        
        # Agent registry
        self.agents = {
            "research": self.research_agent,
            "generation": self.generation_agent,
            "validation": self.validation_agent
        }
        
        self.logger.info("Agent Orchestrator initialized with sub-agents")
    
    def _create_default_config(self) -> AgentConfig:
        """Create default configuration for Agent Orchestrator."""
        capabilities = [
            AgentCapability(
                name="ergoscript_workflow",
                description="Orchestrate complete ErgoScript generation workflow",
                supported_task_types=[TaskType.CODE_GENERATION],
                required_tools=["research_agent", "generation_agent", "validation_agent"]
            ),
            AgentCapability(
                name="multi_agent_coordination",
                description="Coordinate multiple agents for complex tasks",
                supported_task_types=[
                    TaskType.RESEARCH_QUERY,
                    TaskType.CODE_GENERATION,
                    TaskType.CODE_VALIDATION
                ],
                required_tools=["agent_communication", "workflow_management"]
            )
        ]
        
        return AgentConfig(
            agent_id="orchestrator_001",
            role=AgentRole.ORCHESTRATOR,
            name="Agent Orchestrator",
            description="Orchestrates multiple agents for comprehensive task execution",
            capabilities=capabilities,
            max_concurrent_tasks=5,
            timeout_seconds=600  # Longer timeout for complex workflows
        )
    
    async def _execute_task_impl(
        self,
        task_type: TaskType,
        content: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute orchestrated tasks."""
        if task_type == TaskType.CODE_GENERATION:
            return await self._handle_ergoscript_workflow(content, context, metadata)
        elif task_type == TaskType.RESEARCH_QUERY:
            return await self._delegate_to_agent("research", task_type, content, context, metadata)
        elif task_type == TaskType.CODE_VALIDATION:
            return await self._delegate_to_agent("validation", task_type, content, context, metadata)
        else:
            raise ValueError(f"Unsupported task type for Orchestrator: {task_type}")
    
    async def _handle_ergoscript_workflow(
        self,
        requirements: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ErgoScriptResponse:
        """
        Handle complete ErgoScript generation workflow.
        
        This orchestrates the full workflow:
        1. Research relevant documentation and examples
        2. Generate ErgoScript code
        3. Validate the generated code
        4. Provide comprehensive response
        """
        self.logger.info("Starting ErgoScript generation workflow")
        correlation_id = uuid4()
        
        try:
            # Phase 1: Research
            self.logger.info("Phase 1: Research and information gathering")
            research_result = await self._research_phase(
                requirements, context, metadata, correlation_id
            )
            
            # Phase 2: Generation
            self.logger.info("Phase 2: Code generation")
            enhanced_context = self._enhance_context_with_research(
                context, research_result
            )
            generation_result = await self._generation_phase(
                requirements, enhanced_context, metadata, correlation_id
            )
            
            # Phase 3: Validation
            self.logger.info("Phase 3: Code validation")
            validation_result = await self._validation_phase(
                generation_result.generated_code, enhanced_context, metadata, correlation_id
            )
            
            # Phase 4: Final assembly
            self.logger.info("Phase 4: Assembling final response")
            final_response = self._assemble_final_response(
                generation_result, validation_result, research_result
            )
            
            self.logger.info("ErgoScript workflow completed successfully")
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error in ErgoScript workflow: {e}")
            raise
    
    async def _research_phase(
        self,
        requirements: str,
        context: Optional[ConversationContext],
        metadata: Optional[Dict[str, Any]],
        correlation_id
    ) -> Dict[str, Any]:
        """Execute research phase."""
        try:
            # Create research query
            research_query = f"ErgoScript examples and documentation for: {requirements}"
            
            # Execute research task
            research_task_result = await self.research_agent.execute_task(
                task_type=TaskType.RESEARCH_QUERY,
                content=research_query,
                context=context,
                metadata=metadata
            )
            
            if research_task_result.success:
                return research_task_result.result
            else:
                self.logger.warning(f"Research phase failed: {research_task_result.error_message}")
                return {"findings": "No research data available", "sources": "", "recommendations": ""}
                
        except Exception as e:
            self.logger.warning(f"Research phase error: {e}")
            return {"findings": "Research unavailable", "sources": "", "recommendations": ""}
    
    async def _generation_phase(
        self,
        requirements: str,
        context: Optional[ConversationContext],
        metadata: Optional[Dict[str, Any]],
        correlation_id
    ) -> ErgoScriptResponse:
        """Execute generation phase."""
        generation_task_result = await self.generation_agent.execute_task(
            task_type=TaskType.CODE_GENERATION,
            content=requirements,
            context=context,
            metadata=metadata
        )
        
        if generation_task_result.success:
            return generation_task_result.result
        else:
            raise Exception(f"Generation phase failed: {generation_task_result.error_message}")
    
    async def _validation_phase(
        self,
        code: str,
        context: Optional[ConversationContext],
        metadata: Optional[Dict[str, Any]],
        correlation_id
    ) -> Dict[str, Any]:
        """Execute validation phase."""
        try:
            validation_task_result = await self.validation_agent.execute_task(
                task_type=TaskType.CODE_VALIDATION,
                content=code,
                context=context,
                metadata=metadata
            )
            
            if validation_task_result.success:
                return validation_task_result.result
            else:
                self.logger.warning(f"Validation phase failed: {validation_task_result.error_message}")
                return {"is_valid": False, "errors": "Validation unavailable"}
                
        except Exception as e:
            self.logger.warning(f"Validation phase error: {e}")
            return {"is_valid": False, "errors": str(e)}
    
    def _enhance_context_with_research(
        self,
        original_context: Optional[ConversationContext],
        research_result: Dict[str, Any]
    ) -> Optional[ConversationContext]:
        """Enhance context with research findings."""
        if original_context is None:
            original_context = ConversationContext(
                session_id=uuid4(),
                conversation_history=[],
                context_data={}
            )
        
        # Add research findings to context
        original_context.context_data.update({
            "research_findings": research_result.get("findings", ""),
            "relevant_sources": research_result.get("sources", ""),
            "recommendations": research_result.get("recommendations", "")
        })
        
        return original_context
    
    def _assemble_final_response(
        self,
        generation_result: ErgoScriptResponse,
        validation_result: Dict[str, Any],
        research_result: Dict[str, Any]
    ) -> ErgoScriptResponse:
        """Assemble the final comprehensive response."""
        # Update generation result with validation and research data
        generation_result.validation_results = validation_result
        
        # Add research sources to documentation links
        if "sources" in research_result:
            # Could parse sources and add to documentation_links
            pass
        
        # Update metadata with orchestration info
        if not generation_result.metadata:
            generation_result.metadata = {}
        
        generation_result.metadata.update({
            "orchestrated": True,
            "research_conducted": bool(research_result.get("findings")),
            "validation_performed": bool(validation_result),
            "workflow_version": "1.0"
        })
        
        return generation_result
    
    async def _delegate_to_agent(
        self,
        agent_name: str,
        task_type: TaskType,
        content: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Delegate task to specific agent."""
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        agent = self.agents[agent_name]
        task_result = await agent.execute_task(
            task_type=task_type,
            content=content,
            context=context,
            metadata=metadata
        )
        
        if task_result.success:
            return task_result.result
        else:
            raise Exception(f"Agent {agent_name} failed: {task_result.error_message}")
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all managed agents."""
        status = {
            "orchestrator": self.get_status(),
            "agents": {}
        }
        
        for agent_name, agent in self.agents.items():
            status["agents"][agent_name] = agent.get_status()
        
        return status
    
    async def shutdown_all_agents(self) -> None:
        """Shutdown all managed agents."""
        self.logger.info("Shutting down all agents")
        
        shutdown_tasks = [agent.shutdown() for agent in self.agents.values()]
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        await self.shutdown() 