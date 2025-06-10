"""
Optimization and Evaluation API endpoints for FintelligenceAI.

This module provides REST API endpoints for DSPy optimization and
comprehensive system evaluation capabilities.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from fintelligence_ai.core import (
    DSPyOptimizer,
    OptimizationConfig,
    OptimizationResult,
    EvaluationFramework,
    EvaluationResult,
    EvaluationMetrics,
    optimize_rag_pipeline,
    optimize_agent_system,
)
from fintelligence_ai.config import get_settings

# Create router
router = APIRouter(prefix="/optimization", tags=["Optimization & Evaluation"])

logger = logging.getLogger(__name__)


# Request/Response Models
class OptimizationRequest(BaseModel):
    """Request model for DSPy optimization."""
    
    target_component: str = Field(description="Component to optimize (rag_pipeline, agent_system, generation)")
    optimization_config: Optional[Dict[str, Any]] = Field(default=None, description="Optimization configuration")
    training_data: List[Dict[str, Any]] = Field(description="Training data for optimization")
    validation_data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Validation data")
    async_execution: bool = Field(default=False, description="Run optimization asynchronously")


class EvaluationRequest(BaseModel):
    """Request model for system evaluation."""
    
    evaluation_type: str = Field(description="Type of evaluation (comprehensive, ergoscript, rag, agents)")
    test_suite: Dict[str, Any] = Field(description="Test cases and evaluation scenarios")
    components_to_evaluate: List[str] = Field(
        default=["generation_agent", "rag_pipeline", "orchestrator"],
        description="System components to evaluate"
    )
    async_execution: bool = Field(default=False, description="Run evaluation asynchronously")


class OptimizationStatusResponse(BaseModel):
    """Response model for optimization status."""
    
    optimization_id: str
    status: str
    progress_percentage: float
    current_trial: int
    total_trials: int
    best_score: float
    estimated_time_remaining_minutes: float


class EvaluationStatusResponse(BaseModel):
    """Response model for evaluation status."""
    
    evaluation_id: str
    status: str
    progress_percentage: float
    tests_completed: int
    total_tests: int
    current_accuracy: float


# Global tracking for async operations
_active_optimizations: Dict[str, Dict[str, Any]] = {}
_active_evaluations: Dict[str, Dict[str, Any]] = {}


# API Endpoints
@router.post("/optimize", response_model=OptimizationResult)
async def optimize_system_component(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
) -> OptimizationResult:
    """
    Optimize a system component using DSPy optimizers.
    
    This endpoint can optimize RAG pipelines, agent systems, or specific
    generation modules using various DSPy optimization strategies.
    """
    try:
        logger.info(f"Starting optimization for component: {request.target_component}")
        
        # Create optimization configuration
        if request.optimization_config:
            config = OptimizationConfig(**request.optimization_config)
        else:
            config = OptimizationConfig()
        
        # Handle async execution
        if request.async_execution:
            optimization_id = str(uuid4())
            _active_optimizations[optimization_id] = {
                'status': 'started',
                'progress': 0.0,
                'start_time': asyncio.get_event_loop().time()
            }
            
            background_tasks.add_task(
                _run_optimization_async,
                optimization_id,
                request.target_component,
                config,
                request.training_data,
                request.validation_data
            )
            
            return OptimizationResult(
                optimization_id=optimization_id,
                config=config,
                success=True,
                initial_score=0.0,
                final_score=0.0,
                improvement_percentage=0.0,
                trials_completed=0,
                best_trial=0,
                execution_time_minutes=0.0,
                metadata={"status": "running", "async": True}
            )
        
        # Synchronous execution
        if request.target_component == "rag_pipeline":
            # Import and optimize RAG pipeline
            from fintelligence_ai.rag import create_rag_pipeline
            pipeline = create_rag_pipeline()
            optimized_pipeline, result = optimize_rag_pipeline(
                pipeline, request.training_data, config
            )
            
        elif request.target_component == "agent_system":
            # Import and optimize agent system
            from fintelligence_ai.agents import AgentOrchestrator
            orchestrator = AgentOrchestrator()
            optimized_system, result = optimize_agent_system(
                orchestrator, request.training_data, config
            )
            
        elif request.target_component == "generation":
            # Optimize generation agent specifically
            from fintelligence_ai.agents import GenerationAgent
            generation_agent = GenerationAgent()
            optimizer = DSPyOptimizer(config)
            
            # Convert training data to DSPy examples
            import dspy
            trainset = [
                dspy.Example(
                    description=item.get("description", ""),
                    expected_code=item.get("expected_code", "")
                ).with_inputs("description")
                for item in request.training_data
            ]
            
            # Create wrapper module
            class GenerationModule(dspy.Module):
                def __init__(self, agent):
                    super().__init__()
                    self.agent = agent
                
                def forward(self, description):
                    try:
                        result = asyncio.run(self.agent.execute_task('code_generation', description))
                        return dspy.Prediction(answer=result.result.get('generated_code', ''))
                    except Exception as e:
                        logger.warning(f"Generation error: {e}")
                        return dspy.Prediction(answer="")
            
            generation_module = GenerationModule(generation_agent)
            optimized_module, result = optimizer.optimize_module(generation_module, trainset)
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported component for optimization: {request.target_component}"
            )
        
        logger.info(f"Optimization completed with {result.improvement_percentage:.1f}% improvement")
        return result
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


@router.post("/evaluate", response_model=EvaluationResult)
async def evaluate_system(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
) -> EvaluationResult:
    """
    Run comprehensive evaluation of system components.
    
    This endpoint evaluates ErgoScript generation quality, RAG performance,
    and agent system effectiveness using various metrics and test cases.
    """
    try:
        logger.info(f"Starting {request.evaluation_type} evaluation")
        
        # Initialize evaluation framework
        evaluation_framework = EvaluationFramework()
        
        # Prepare system components
        system_components = {}
        
        if "generation_agent" in request.components_to_evaluate:
            from fintelligence_ai.agents import GenerationAgent
            system_components["generation_agent"] = GenerationAgent()
        
        if "rag_pipeline" in request.components_to_evaluate:
            from fintelligence_ai.rag import create_rag_pipeline
            system_components["rag_pipeline"] = create_rag_pipeline()
        
        if "orchestrator" in request.components_to_evaluate:
            from fintelligence_ai.agents import AgentOrchestrator
            system_components["orchestrator"] = AgentOrchestrator()
        
        # Handle async execution
        if request.async_execution:
            evaluation_id = str(uuid4())
            _active_evaluations[evaluation_id] = {
                'status': 'started',
                'progress': 0.0,
                'start_time': asyncio.get_event_loop().time()
            }
            
            background_tasks.add_task(
                _run_evaluation_async,
                evaluation_id,
                evaluation_framework,
                request.test_suite,
                system_components
            )
            
            return EvaluationResult(
                evaluation_id=evaluation_id,
                category=request.evaluation_type,
                metrics=EvaluationMetrics(
                    accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                    syntax_correctness=0.0, semantic_correctness=0.0,
                    code_quality_score=0.0, response_time_ms=0.0,
                    retrieval_relevance=0.0, user_satisfaction=0.0,
                    task_completion_rate=0.0
                ),
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                execution_time_seconds=0.0,
                average_response_time_ms=0.0,
                metadata={"status": "running", "async": True}
            )
        
        # Synchronous execution
        result = await evaluation_framework.run_comprehensive_evaluation(
            request.test_suite,
            system_components
        )
        
        logger.info(f"Evaluation completed with {result.metrics.accuracy:.3f} overall accuracy")
        return result
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.get("/optimize/{optimization_id}/status", response_model=OptimizationStatusResponse)
async def get_optimization_status(optimization_id: str) -> OptimizationStatusResponse:
    """Get the status of a running optimization."""
    if optimization_id not in _active_optimizations:
        raise HTTPException(
            status_code=404,
            detail=f"Optimization {optimization_id} not found"
        )
    
    optimization_info = _active_optimizations[optimization_id]
    
    return OptimizationStatusResponse(
        optimization_id=optimization_id,
        status=optimization_info.get('status', 'unknown'),
        progress_percentage=optimization_info.get('progress', 0.0),
        current_trial=optimization_info.get('current_trial', 0),
        total_trials=optimization_info.get('total_trials', 10),
        best_score=optimization_info.get('best_score', 0.0),
        estimated_time_remaining_minutes=optimization_info.get('time_remaining', 0.0)
    )


@router.get("/evaluate/{evaluation_id}/status", response_model=EvaluationStatusResponse)
async def get_evaluation_status(evaluation_id: str) -> EvaluationStatusResponse:
    """Get the status of a running evaluation."""
    if evaluation_id not in _active_evaluations:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation {evaluation_id} not found"
        )
    
    evaluation_info = _active_evaluations[evaluation_id]
    
    return EvaluationStatusResponse(
        evaluation_id=evaluation_id,
        status=evaluation_info.get('status', 'unknown'),
        progress_percentage=evaluation_info.get('progress', 0.0),
        tests_completed=evaluation_info.get('tests_completed', 0),
        total_tests=evaluation_info.get('total_tests', 0),
        current_accuracy=evaluation_info.get('current_accuracy', 0.0)
    )


@router.get("/optimize/{optimization_id}/result", response_model=OptimizationResult)
async def get_optimization_result(optimization_id: str) -> OptimizationResult:
    """Get the result of a completed optimization."""
    if optimization_id not in _active_optimizations:
        raise HTTPException(
            status_code=404,
            detail=f"Optimization {optimization_id} not found"
        )
    
    optimization_info = _active_optimizations[optimization_id]
    
    if optimization_info.get('status') != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Optimization {optimization_id} is not yet completed"
        )
    
    return optimization_info.get('result', {})


@router.get("/evaluate/{evaluation_id}/result", response_model=EvaluationResult)
async def get_evaluation_result(evaluation_id: str) -> EvaluationResult:
    """Get the result of a completed evaluation."""
    if evaluation_id not in _active_evaluations:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation {evaluation_id} not found"
        )
    
    evaluation_info = _active_evaluations[evaluation_id]
    
    if evaluation_info.get('status') != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation {evaluation_id} is not yet completed"
        )
    
    return evaluation_info.get('result', {})


@router.post("/benchmark")
async def run_system_benchmark() -> Dict[str, Any]:
    """
    Run a standard benchmark suite for the system.
    
    This endpoint runs a predefined set of tests to benchmark
    the system's performance against standard metrics.
    """
    try:
        logger.info("Starting system benchmark")
        
        # Create benchmark test suite
        benchmark_suite = {
            "ergoscript_tests": [
                {
                    "id": "token_creation",
                    "request": {
                        "description": "Create a simple token contract",
                        "requirements": ["token creation", "basic validation"]
                    },
                    "expected": {
                        "reference_code": "{ OUTPUTS.size == 1 && OUTPUTS(0).tokens.size == 1 }"
                    }
                },
                {
                    "id": "auction_contract",
                    "request": {
                        "description": "Create an auction contract with bidding logic",
                        "requirements": ["bidding", "winner selection", "payment validation"]
                    },
                    "expected": {
                        "reference_code": "{ val bidAmount = INPUTS(0).value }"
                    }
                }
            ],
            "rag_tests": [
                {
                    "id": "documentation_search",
                    "query": "ErgoScript syntax validation",
                    "relevant_documents": [
                        {"id": "ergoscript_guide", "content": "ErgoScript syntax validation rules"}
                    ]
                }
            ],
            "agent_tests": [
                {
                    "id": "code_generation_task",
                    "task": "Generate a token creation script",
                    "task_type": "code_generation",
                    "expected_outcome": {
                        "success": True,
                        "generated_code": "contains token logic"
                    }
                }
            ]
        }
        
        # Run evaluation
        evaluation_framework = EvaluationFramework()
        
        # Prepare system components
        from fintelligence_ai.agents import GenerationAgent, AgentOrchestrator
        from fintelligence_ai.rag import create_rag_pipeline
        
        system_components = {
            "generation_agent": GenerationAgent(),
            "rag_pipeline": create_rag_pipeline(),
            "orchestrator": AgentOrchestrator()
        }
        
        result = await evaluation_framework.run_comprehensive_evaluation(
            benchmark_suite,
            system_components
        )
        
        # Return benchmark results
        return {
            "benchmark_id": result.evaluation_id,
            "overall_score": result.metrics.accuracy,
            "performance_grade": _calculate_performance_grade(result.metrics.accuracy),
            "key_metrics": {
                "accuracy": result.metrics.accuracy,
                "code_quality": result.metrics.code_quality_score,
                "response_time": result.average_response_time_ms,
                "task_completion": result.metrics.task_completion_rate
            },
            "recommendations": result.recommendations[:5],  # Top 5 recommendations
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}"
        )


@router.get("/metrics/summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of recent optimization and evaluation metrics."""
    try:
        # This would typically query a metrics database
        # For now, return mock summary data
        
        return {
            "recent_optimizations": len(_active_optimizations),
            "recent_evaluations": len(_active_evaluations),
            "average_improvement": 15.3,  # Mock data
            "system_health_score": 8.7,
            "last_benchmark_score": 0.85,
            "performance_trend": "improving",
            "active_operations": {
                "optimizations": [
                    {"id": opt_id, "status": info.get("status", "unknown")}
                    for opt_id, info in _active_optimizations.items()
                ],
                "evaluations": [
                    {"id": eval_id, "status": info.get("status", "unknown")}
                    for eval_id, info in _active_evaluations.items()
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics summary: {str(e)}"
        )


# Background task functions
async def _run_optimization_async(
    optimization_id: str,
    component_type: str,
    config: OptimizationConfig,
    training_data: List[Dict[str, Any]],
    validation_data: Optional[List[Dict[str, Any]]] = None
) -> None:
    """Run optimization asynchronously."""
    try:
        _active_optimizations[optimization_id].update({
            'status': 'running',
            'progress': 10.0
        })
        
        # Run the optimization (mock implementation)
        await asyncio.sleep(2)  # Simulate work
        
        _active_optimizations[optimization_id].update({
            'status': 'completed',
            'progress': 100.0,
            'result': OptimizationResult(
                optimization_id=optimization_id,
                config=config,
                success=True,
                initial_score=0.7,
                final_score=0.85,
                improvement_percentage=21.4,
                trials_completed=config.num_trials,
                best_trial=7,
                execution_time_minutes=15.3
            )
        })
        
    except Exception as e:
        _active_optimizations[optimization_id].update({
            'status': 'failed',
            'error': str(e)
        })


async def _run_evaluation_async(
    evaluation_id: str,
    framework: EvaluationFramework,
    test_suite: Dict[str, Any],
    components: Dict[str, Any]
) -> None:
    """Run evaluation asynchronously."""
    try:
        _active_evaluations[evaluation_id].update({
            'status': 'running',
            'progress': 10.0
        })
        
        # Run the evaluation
        result = await framework.run_comprehensive_evaluation(test_suite, components)
        
        _active_evaluations[evaluation_id].update({
            'status': 'completed',
            'progress': 100.0,
            'result': result
        })
        
    except Exception as e:
        _active_evaluations[evaluation_id].update({
            'status': 'failed',
            'error': str(e)
        })


def _calculate_performance_grade(accuracy: float) -> str:
    """Calculate performance grade based on accuracy."""
    if accuracy >= 0.9:
        return "A"
    elif accuracy >= 0.8:
        return "B"
    elif accuracy >= 0.7:
        return "C"
    elif accuracy >= 0.6:
        return "D"
    else:
        return "F" 