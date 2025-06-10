"""
FintelligenceAI Core Module

This module provides core functionality including optimization, evaluation,
and advanced DSPy integration for the FintelligenceAI system.
"""

from .evaluation import (
    AgentEvaluator,
    ErgoScriptEvaluator,
    EvaluationFramework,
    EvaluationMetrics,
    EvaluationResult,
    RAGEvaluator,
)
from .optimization import (
    DSPyOptimizer,
    OptimizationConfig,
    OptimizationResult,
    optimize_agent_system,
    optimize_rag_pipeline,
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
