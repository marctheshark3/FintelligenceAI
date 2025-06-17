"""
Generation Agent for FintelligenceAI.

This module implements the GenerationAgent that specializes in ErgoScript code
generation using DSPy modules and RAG-enhanced context.
"""

from typing import Any, Optional

import dspy

from .base import BaseAgent
from .types import (
    AgentCapability,
    AgentConfig,
    AgentRole,
    ConversationContext,
    ErgoScriptRequest,
    ErgoScriptResponse,
    TaskType,
)


class CodeGeneration(dspy.Signature):
    """Generate code based on requirements and research context."""

    description: str = dspy.InputField(
        desc="Natural language description of the required code"
    )
    requirements: str = dspy.InputField(desc="Specific requirements and constraints")
    context: str = dspy.InputField(
        desc="CRITICAL: Research findings with correct API usage patterns. MUST follow the exact imports and method signatures provided in this context."
    )
    use_case: str = dspy.InputField(desc="Specific use case (token, auction, etc.)")

    code: str = dspy.OutputField(
        desc="Generated code using patterns and methods from the context"
    )
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
                required_tools=["dspy_modules"],
            )
        ]

        return AgentConfig(
            agent_id="generation_agent_001",
            role=AgentRole.GENERATION,
            name="Generation Agent",
            description="Specialized agent for ErgoScript code generation",
            capabilities=capabilities,
            max_concurrent_tasks=2,
            timeout_seconds=300,
        )

    def _initialize_dspy_modules(self) -> None:
        """Initialize DSPy modules for code generation tasks."""
        self.dspy_modules.update(
            {
                "code_generator": dspy.ChainOfThought(CodeGeneration),
            }
        )

    async def _execute_task_impl(
        self,
        task_type: TaskType,
        content: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[dict[str, Any]] = None,
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
        metadata: Optional[dict[str, Any]] = None,
    ) -> ErgoScriptResponse:
        """Handle ErgoScript code generation requests."""
        self.logger.info("Generating ErgoScript code")

        try:
            # Create request
            request = ErgoScriptRequest(
                description=requirements,
                use_case=metadata.get("use_case", "general") if metadata else "general",
            )

            # Extract context from research findings
            context_str = ""
            if context and context.context_data:
                research_findings = context.context_data.get("research_findings", "")
                relevant_sources = context.context_data.get("relevant_sources", "")
                recommendations = context.context_data.get("recommendations", "")

                if research_findings:
                    context_str += f"CRITICAL RESEARCH FINDINGS - MUST FOLLOW EXACTLY:\n{research_findings}\n\n"
                if relevant_sources:
                    context_str += f"AUTHORITATIVE SOURCES:\n{relevant_sources}\n\n"
                if recommendations:
                    context_str += f"MANDATORY RECOMMENDATIONS:\n{recommendations}\n\n"

                # Add explicit instructions
                context_str += "IMPORTANT: You MUST use the EXACT imports and method signatures shown in the research findings above. Do NOT use any other imports or methods not mentioned in the research context.\n\n"

                self.logger.info(
                    f"Using enhanced context with {len(context_str)} characters of research data"
                )

            # Generate code using direct OpenAI API if we have research context
            if (
                context_str and len(context_str) > 100
            ):  # We have meaningful research context
                self.logger.info(
                    "Using direct OpenAI API with research context for better accuracy"
                )
                result = await self._generate_with_research_context(
                    description=request.description,
                    requirements=requirements,
                    context=context_str,
                    use_case=request.use_case,
                )
            else:
                # Fallback to DSPy if no research context
                self.logger.info("Using DSPy module (no research context available)")
                generator = self.dspy_modules["code_generator"]
                result = generator(
                    description=request.description,
                    requirements=requirements,
                    context=context_str,
                    use_case=request.use_case,
                )

            # Create response
            response = ErgoScriptResponse(
                generated_code=result.code,
                explanation=result.explanation,
                complexity_score=result.complexity_score,
                validation_results={},
            )

            return response

        except Exception as e:
            self.logger.error(f"Error in code generation: {e}")
            raise

    def _analyze_content_type(self, context: str) -> dict:
        """Analyze the retrieved content to determine the type and patterns."""
        analysis = {
            "language": "unknown",
            "type": "unknown",
            "imports": [],
            "key_patterns": [],
            "examples": [],
        }

        context_lower = context.lower()

        # Detect programming language
        if "function " in context and (
            "javascript" in context_lower
            or "const " in context
            or "let " in context
            or "var " in context
        ):
            analysis["language"] = "javascript"
        elif "import " in context and (
            "python" in context_lower or ".py" in context_lower
        ):
            analysis["language"] = "python"
        elif "contract" in context_lower and (
            "solidity" in context_lower or "pragma" in context_lower
        ):
            analysis["language"] = "solidity"
        elif "ergoscript" in context_lower or "ergo" in context_lower:
            analysis["language"] = "ergoscript"
        elif "rust" in context_lower or "cargo" in context_lower:
            analysis["language"] = "rust"

        # Detect content type
        if "api" in context_lower and (
            "endpoint" in context_lower or "http" in context_lower
        ):
            analysis["type"] = "api"
        elif (
            "library" in context_lower
            or "package" in context_lower
            or "import" in context
        ):
            analysis["type"] = "library"
        elif "smart contract" in context_lower or "contract" in context_lower:
            analysis["type"] = "smart_contract"
        elif "cli" in context_lower or "command" in context_lower:
            analysis["type"] = "cli"

        # Extract import patterns
        import_lines = [
            line.strip() for line in context.split("\n") if "import " in line
        ]
        analysis["imports"] = import_lines[:5]  # Take first 5 imports

        # Extract code examples
        code_blocks = []
        lines = context.split("\n")
        in_code_block = False
        current_block = []

        for line in lines:
            if "```" in line:
                if in_code_block:
                    if current_block:
                        code_blocks.append("\n".join(current_block))
                    current_block = []
                    in_code_block = False
                else:
                    in_code_block = True
            elif in_code_block:
                current_block.append(line)

        analysis["examples"] = code_blocks[:3]  # Take first 3 code examples

        return analysis

    def _build_dynamic_prompt(
        self,
        description: str,
        requirements: str,
        context: str,
        use_case: str,
        analysis: dict,
    ) -> str:
        """Build a dynamic prompt based on content analysis."""

        # Base prompt structure
        prompt = f"""You are an expert developer. You MUST follow the provided documentation EXACTLY.

CONTENT ANALYSIS:
- Language: {analysis['language']}
- Type: {analysis['type']}

"""

        # Add language-specific guidance
        if analysis["language"] == "python":
            prompt += "CRITICAL: This is Python code. Follow Python syntax and conventions.\n\n"
            if analysis["imports"]:
                prompt += "CORRECT IMPORTS (from documentation):\n"
                for imp in analysis["imports"]:
                    prompt += f"```python\n{imp}\n```\n"
                prompt += "\n"
        elif analysis["language"] == "javascript":
            prompt += "CRITICAL: This is JavaScript/Node.js code. Follow JavaScript syntax and conventions.\n\n"
        elif analysis["language"] == "solidity":
            prompt += "CRITICAL: This is Solidity smart contract code. Follow Solidity syntax and conventions.\n\n"
        elif analysis["language"] == "ergoscript":
            prompt += "CRITICAL: This is ErgoScript code. Follow ErgoScript syntax and conventions.\n\n"
        elif analysis["language"] == "rust":
            prompt += (
                "CRITICAL: This is Rust code. Follow Rust syntax and conventions.\n\n"
            )

        # Add examples from the documentation
        if analysis["examples"]:
            prompt += "EXAMPLES FROM DOCUMENTATION:\n"
            for i, example in enumerate(
                analysis["examples"][:2]
            ):  # Limit to 2 examples
                prompt += f"Example {i+1}:\n```\n{example}\n```\n\n"

        # Add the research context
        prompt += f"COMPLETE DOCUMENTATION:\n{context}\n\n"

        # Add the task details
        prompt += f"""TASK: {description}
REQUIREMENTS: {requirements}
USE CASE: {use_case}

MANDATORY RULES:
1. Use ONLY the patterns, imports, and methods shown in the documentation above
2. Do NOT use any APIs, imports, or methods not mentioned in the documentation
3. Follow the EXACT syntax and patterns from the examples
4. Include proper error handling
5. Add clear comments explaining each step

Generate complete, working code that follows the documentation exactly."""

        return prompt

    async def _generate_with_research_context(
        self,
        description: str,
        requirements: str,
        context: str,
        use_case: str,
    ) -> Any:
        """Generate code using direct OpenAI API with research context."""
        import os

        import openai

        # Analyze the content to understand what we're working with
        content_analysis = self._analyze_content_type(context)

        # Debug: Log the content analysis
        self.logger.info(f"Content analysis: {content_analysis}")
        self.logger.info(f"Context preview (first 200 chars): {context[:200]}...")

        # Create a dynamic prompt based on the content analysis
        prompt = self._build_dynamic_prompt(
            description, requirements, context, use_case, content_analysis
        )

        try:
            # Debug: Log the prompt being sent
            self.logger.info(
                f"Direct OpenAI prompt (first 500 chars): {prompt[:500]}..."
            )
            self.logger.info(f"Full prompt length: {len(prompt)} characters")

            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Python developer who follows provided API documentation exactly. You MUST use only the imports and methods shown in the user's research context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=2000,
            )

            generated_code = response.choices[0].message.content

            # Create a result object that matches DSPy output format
            class DirectResult:
                def __init__(self, code, explanation, complexity):
                    self.code = code
                    self.explanation = explanation
                    self.complexity_score = complexity

            # Extract just the code part if it's wrapped in markdown
            if "```python" in generated_code:
                code_start = generated_code.find("```python") + 9
                code_end = generated_code.find("```", code_start)
                if code_end != -1:
                    code = generated_code[code_start:code_end].strip()
                else:
                    code = generated_code[code_start:].strip()
            else:
                code = generated_code

            return DirectResult(
                code=code,
                explanation=f"Generated using research context with {len(context)} characters of API documentation",
                complexity=5.0,
            )

        except Exception as e:
            self.logger.error(f"Error in direct OpenAI generation: {e}")
            # Fallback to DSPy
            generator = self.dspy_modules["code_generator"]
            return generator(
                description=description,
                requirements=requirements,
                context=context,
                use_case=use_case,
            )
