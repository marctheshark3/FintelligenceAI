"""
FintelligenceAI Core Module

This module provides core functionality including optimization, evaluation,
and advanced DSPy integration for the FintelligenceAI system.
"""

from .optimization import (
    DSPyOptimizer,
    OptimizationConfig,
    OptimizationResult,
    optimize_rag_pipeline,
    optimize_agent_system,
)
from .evaluation import (
    EvaluationFramework,
    EvaluationMetrics,
    EvaluationResult,
    ErgoScriptEvaluator,
    RAGEvaluator,
    AgentEvaluator,
)

__all__ = [
    "DSPyOptimizer",
    "OptimizationConfig", 
    "OptimizationResult",
    "optimize_rag_pipeline",
    "optimize_agent_system",
    "EvaluationFramework",
    "EvaluationMetrics",
    "EvaluationResult", 
    "ErgoScriptEvaluator",
    "RAGEvaluator",
    "AgentEvaluator",
]
