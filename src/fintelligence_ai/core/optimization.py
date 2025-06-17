"""
DSPy Optimization Framework for FintelligenceAI

This module provides comprehensive optimization capabilities using DSPy optimizers
including MIPROv2, BootstrapFinetune, and others for RAG pipelines and agent systems.
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import dspy
from pydantic import BaseModel, Field

from fintelligence_ai.config import get_settings

logger = logging.getLogger(__name__)


class OptimizerType(str, Enum):
    """Supported DSPy optimizer types."""

    MIPROV2 = "miprov2"
    BOOTSTRAP_FINETUNE = "bootstrap_finetune"
    COPRO = "copro"
    TELEPROMPT = "teleprompt"
    SIGNATURE_OPT = "signature_opt"


class OptimizationStrategy(str, Enum):
    """Optimization strategies for different components."""

    ACCURACY_FOCUSED = "accuracy_focused"
    SPEED_FOCUSED = "speed_focused"
    BALANCED = "balanced"
    COST_OPTIMIZED = "cost_optimized"
    ERGOSCRIPT_SPECIALIZED = "ergoscript_specialized"


class OptimizationConfig(BaseModel):
    """Configuration for DSPy optimization."""

    optimizer_type: OptimizerType = Field(
        default=OptimizerType.MIPROV2, description="Type of optimizer to use"
    )
    strategy: OptimizationStrategy = Field(
        default=OptimizationStrategy.BALANCED, description="Optimization strategy"
    )
    num_trials: int = Field(default=10, description="Number of optimization trials")
    max_bootstraps: int = Field(default=20, description="Maximum bootstrap examples")
    max_labeled_demos: int = Field(
        default=5, description="Maximum labeled demonstrations"
    )
    timeout_minutes: int = Field(
        default=60, description="Optimization timeout in minutes"
    )
    metric_threshold: float = Field(default=0.8, description="Target metric threshold")

    # Component-specific settings
    optimize_retrieval: bool = Field(
        default=True, description="Optimize retrieval components"
    )
    optimize_generation: bool = Field(
        default=True, description="Optimize generation components"
    )
    optimize_reranking: bool = Field(
        default=True, description="Optimize reranking components"
    )
    optimize_agents: bool = Field(default=True, description="Optimize agent components")

    # Advanced settings
    use_multi_stage: bool = Field(
        default=True, description="Use multi-stage optimization"
    )
    save_intermediate: bool = Field(
        default=True, description="Save intermediate results"
    )
    parallel_execution: bool = Field(
        default=False, description="Use parallel optimization"
    )

    # Model configuration
    llm_config: dict[str, Any] = Field(
        default_factory=dict, description="Model configuration overrides"
    )
    evaluation_model: Optional[str] = Field(
        default=None, description="Model for evaluation"
    )


class OptimizationResult(BaseModel):
    """Result of an optimization run."""

    optimization_id: str = Field(description="Unique optimization identifier")
    config: OptimizationConfig = Field(description="Optimization configuration used")
    success: bool = Field(description="Whether optimization completed successfully")

    # Performance metrics
    initial_score: float = Field(description="Initial performance score")
    final_score: float = Field(description="Final performance score")
    improvement_percentage: float = Field(description="Improvement percentage")

    # Optimization details
    trials_completed: int = Field(description="Number of trials completed")
    best_trial: int = Field(description="Best performing trial number")
    execution_time_minutes: float = Field(description="Total execution time")

    # Component results
    component_results: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Per-component results"
    )
    optimized_prompts: dict[str, str] = Field(
        default_factory=dict, description="Optimized prompts"
    )
    optimized_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Optimized parameters"
    )

    # Metadata
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Optimization timestamp"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class DSPyOptimizer:
    """Main optimizer class for DSPy components."""

    def __init__(self, config: OptimizationConfig):
        """Initialize the optimizer with configuration."""
        self.config = config
        self.settings = get_settings()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize DSPy with configured model
        self._setup_dspy()

    def _setup_dspy(self) -> None:
        """Setup DSPy with the configured model."""
        try:
            # Configure the primary model
            if self.config.llm_config:
                model_config = self.config.llm_config
            else:
                # Determine default model based on provider and local mode
                if (
                    self.settings.dspy.local_mode
                    or self.settings.dspy.model_provider == "ollama"
                ):
                    model_config = {
                        "model": self.settings.ollama.model,
                        "max_tokens": self.settings.ollama.max_tokens,
                        "temperature": self.settings.ollama.temperature,
                    }
                else:
                    model_config = {
                        "model": self.settings.openai.model,
                        "max_tokens": 1000,
                        "temperature": 0.0,
                    }

            # Set up the language model based on provider
            provider = self.settings.dspy.model_provider.lower()

            if self.settings.dspy.local_mode or provider == "ollama":
                # Use Ollama for local-only mode
                from .ollama import get_ollama_dspy_model

                lm = get_ollama_dspy_model(
                    model=model_config.get("model"),
                    temperature=model_config.get("temperature", 0.1),
                    max_tokens=model_config.get("max_tokens", 4096),
                )
            elif "gpt" in model_config.get("model", "").lower() or provider == "openai":
                # Use the correct DSPy LM interface for OpenAI
                model_name = model_config.get("model", "gpt-4")
                if not model_name.startswith("openai/"):
                    model_name = f"openai/{model_name}"

                lm = dspy.LM(
                    model=model_name,
                    api_key=model_config.get("api_key"),
                    temperature=model_config.get("temperature", 0.0),
                    max_tokens=model_config.get("max_tokens", 1000),
                )
            elif (
                "claude" in model_config.get("model", "").lower()
                or provider == "claude"
            ):
                # Use the correct DSPy LM interface for Claude
                model_name = model_config.get("model", "claude-3-sonnet")
                if not model_name.startswith("anthropic/"):
                    model_name = f"anthropic/{model_name}"

                lm = dspy.LM(
                    model=model_name,
                    api_key=model_config.get("api_key"),
                    temperature=model_config.get("temperature", 0.0),
                    max_tokens=model_config.get("max_tokens", 1000),
                )
            else:
                # Default to OpenAI if no specific provider
                model_name = model_config.get("model", "gpt-4")
                if not model_name.startswith("openai/"):
                    model_name = f"openai/{model_name}"

                lm = dspy.LM(
                    model=model_name,
                    api_key=model_config.get("api_key"),
                    temperature=model_config.get("temperature", 0.0),
                    max_tokens=model_config.get("max_tokens", 1000),
                )

            dspy.configure(lm=lm)
            provider_name = (
                "Ollama (Local)" if self.settings.dspy.local_mode else provider.title()
            )
            self.logger.info(
                f"DSPy configured with {provider_name} - Model: {model_config.get('model')}"
            )

        except Exception as e:
            self.logger.error(f"Failed to setup DSPy: {e}")
            raise

    def optimize_module(
        self,
        module: dspy.Module,
        trainset: list[dspy.Example],
        valset: Optional[list[dspy.Example]] = None,
        metric: Optional[callable] = None,
    ) -> tuple[dspy.Module, OptimizationResult]:
        """
        Optimize a DSPy module using the configured optimizer.

        Args:
            module: DSPy module to optimize
            trainset: Training examples
            valset: Validation examples (optional)
            metric: Evaluation metric function

        Returns:
            Tuple of (optimized_module, optimization_result)
        """
        start_time = time.time()
        optimization_id = f"opt_{int(start_time)}"

        try:
            self.logger.info(
                f"Starting optimization {optimization_id} with {self.config.optimizer_type}"
            )

            # Prepare data
            if valset is None:
                # Split trainset if no validation set provided
                split_idx = int(len(trainset) * 0.8)
                valset = trainset[split_idx:]
                trainset = trainset[:split_idx]

            # Setup metric
            if metric is None:
                metric = self._get_default_metric()

            # Get initial performance
            initial_score = self._evaluate_module(module, valset, metric)
            self.logger.info(f"Initial performance score: {initial_score:.3f}")

            # Setup optimizer
            optimizer = self._create_optimizer()

            # Run optimization
            optimized_module = optimizer.compile(
                student=module,
                trainset=trainset,
                valset=valset,
                **self._get_optimizer_kwargs(),
            )

            # Evaluate optimized module
            final_score = self._evaluate_module(optimized_module, valset, metric)
            improvement = (
                ((final_score - initial_score) / initial_score) * 100
                if initial_score > 0
                else 0
            )

            execution_time = (time.time() - start_time) / 60  # Convert to minutes

            self.logger.info(
                f"Optimization completed. Final score: {final_score:.3f} "
                f"(+{improvement:.1f}% improvement)"
            )

            # Create result
            result = OptimizationResult(
                optimization_id=optimization_id,
                config=self.config,
                success=True,
                initial_score=initial_score,
                final_score=final_score,
                improvement_percentage=improvement,
                trials_completed=self.config.num_trials,
                best_trial=1,  # Would be populated by actual optimizer
                execution_time_minutes=execution_time,
                optimized_prompts=self._extract_prompts(optimized_module),
                optimized_parameters=self._extract_parameters(optimized_module),
            )

            return optimized_module, result

        except Exception as e:
            execution_time = (time.time() - start_time) / 60
            self.logger.error(f"Optimization failed: {e}")

            result = OptimizationResult(
                optimization_id=optimization_id,
                config=self.config,
                success=False,
                initial_score=0.0,
                final_score=0.0,
                improvement_percentage=0.0,
                trials_completed=0,
                best_trial=0,
                execution_time_minutes=execution_time,
                error_message=str(e),
            )

            return module, result

    def _create_optimizer(self):
        """Create the appropriate DSPy optimizer."""
        try:
            if self.config.optimizer_type == OptimizerType.MIPROV2:
                return dspy.MIPROv2(
                    metric=self._get_default_metric(),
                    num_trials=self.config.num_trials,
                    max_bootstraps=self.config.max_bootstraps,
                )
            elif self.config.optimizer_type == OptimizerType.BOOTSTRAP_FINETUNE:
                return dspy.BootstrapFinetune(
                    metric=self._get_default_metric(),
                    max_labeled_demos=self.config.max_labeled_demos,
                )
            elif self.config.optimizer_type == OptimizerType.COPRO:
                return dspy.COPRO(
                    metric=self._get_default_metric(), breadth=self.config.num_trials
                )
            else:
                # Default to MIPROv2
                return dspy.MIPROv2(
                    metric=self._get_default_metric(), num_trials=self.config.num_trials
                )
        except Exception as e:
            self.logger.warning(
                f"Failed to create {self.config.optimizer_type}, using basic optimizer: {e}"
            )
            # Fallback to a simple optimizer
            return dspy.BootstrapFewShot(
                metric=self._get_default_metric(),
                max_labeled_demos=self.config.max_labeled_demos,
            )

    def _get_optimizer_kwargs(self) -> dict[str, Any]:
        """Get optimizer-specific keyword arguments."""
        kwargs = {}

        if self.config.optimizer_type == OptimizerType.MIPROV2:
            kwargs.update(
                {
                    "max_bootstraps": self.config.max_bootstraps,
                    "num_trials": self.config.num_trials,
                }
            )

        return kwargs

    def _get_default_metric(self) -> callable:
        """Get default evaluation metric."""

        def accuracy_metric(gold, pred, trace=None):
            """Default accuracy metric."""
            if hasattr(gold, "answer") and hasattr(pred, "answer"):
                return gold.answer.lower().strip() == pred.answer.lower().strip()
            elif isinstance(gold, str) and isinstance(pred, str):
                return gold.lower().strip() == pred.lower().strip()
            else:
                return str(gold).lower().strip() == str(pred).lower().strip()

        return accuracy_metric

    def _evaluate_module(
        self, module: dspy.Module, dataset: list[dspy.Example], metric: callable
    ) -> float:
        """Evaluate a module on a dataset."""
        if not dataset:
            return 0.0

        correct = 0
        total = len(dataset)

        for example in dataset:
            try:
                pred = module(**example.inputs())
                if metric(example, pred):
                    correct += 1
            except Exception as e:
                self.logger.warning(f"Evaluation error on example: {e}")
                continue

        return correct / total if total > 0 else 0.0

    def _extract_prompts(self, module: dspy.Module) -> dict[str, str]:
        """Extract optimized prompts from a module."""
        prompts = {}

        # This would be more sophisticated in a real implementation
        # to extract actual optimized prompts from the module
        try:
            for name, component in module.named_children():
                if hasattr(component, "signature"):
                    prompts[name] = str(component.signature)
        except Exception as e:
            self.logger.warning(f"Could not extract prompts: {e}")

        return prompts

    def _extract_parameters(self, module: dspy.Module) -> dict[str, Any]:
        """Extract optimized parameters from a module."""
        parameters = {}

        # Extract relevant parameters that were optimized
        try:
            for name, component in module.named_children():
                if hasattr(component, "config"):
                    parameters[name] = component.config
        except Exception as e:
            self.logger.warning(f"Could not extract parameters: {e}")

        return parameters


def optimize_rag_pipeline(
    rag_pipeline,
    training_data: list[dict[str, Any]],
    config: Optional[OptimizationConfig] = None,
) -> tuple[Any, OptimizationResult]:
    """
    Optimize a RAG pipeline using DSPy optimizers.

    Args:
        rag_pipeline: RAG pipeline to optimize
        training_data: Training data for optimization
        config: Optimization configuration

    Returns:
        Tuple of (optimized_pipeline, optimization_result)
    """
    if config is None:
        config = OptimizationConfig(strategy=OptimizationStrategy.BALANCED)

    optimizer = DSPyOptimizer(config)

    # Convert training data to DSPy examples
    trainset = [
        dspy.Example(
            question=item.get("question", ""),
            context=item.get("context", ""),
            answer=item.get("answer", ""),
        ).with_inputs("question", "context")
        for item in training_data
    ]

    # Create a DSPy module wrapper for the RAG pipeline
    class RAGModule(dspy.Module):
        def __init__(self, pipeline):
            super().__init__()
            self.pipeline = pipeline

        def forward(self, question, context=""):
            try:
                result = self.pipeline.generate(question, context)
                return dspy.Prediction(answer=result.get("generated_text", ""))
            except Exception as e:
                logger.warning(f"RAG pipeline error: {e}")
                return dspy.Prediction(answer="")

    rag_module = RAGModule(rag_pipeline)

    return optimizer.optimize_module(rag_module, trainset)


def optimize_agent_system(
    agent_system,
    training_scenarios: list[dict[str, Any]],
    config: Optional[OptimizationConfig] = None,
) -> tuple[Any, OptimizationResult]:
    """
    Optimize an agent system using DSPy optimizers.

    Args:
        agent_system: Agent system to optimize
        training_scenarios: Training scenarios for optimization
        config: Optimization configuration

    Returns:
        Tuple of (optimized_system, optimization_result)
    """
    if config is None:
        config = OptimizationConfig(
            strategy=OptimizationStrategy.ERGOSCRIPT_SPECIALIZED, optimize_agents=True
        )

    optimizer = DSPyOptimizer(config)

    # Convert scenarios to DSPy examples
    trainset = [
        dspy.Example(
            task=scenario.get("task", ""),
            context=scenario.get("context", ""),
            expected_output=scenario.get("expected_output", ""),
        ).with_inputs("task", "context")
        for scenario in training_scenarios
    ]

    # Create DSPy module wrapper for agent system
    class AgentModule(dspy.Module):
        def __init__(self, system):
            super().__init__()
            self.system = system

        def forward(self, task, context=""):
            try:
                result = self.system.execute_task(task, context)
                return dspy.Prediction(answer=result.get("result", ""))
            except Exception as e:
                logger.warning(f"Agent system error: {e}")
                return dspy.Prediction(answer="")

    agent_module = AgentModule(agent_system)

    return optimizer.optimize_module(agent_module, trainset)
