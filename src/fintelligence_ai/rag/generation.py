"""
Generation engine using DSPy modules for RAG-powered text and code generation.

This module implements the generation component of the RAG pipeline using DSPy
modules for consistent and optimizable text generation, with specialized
support for ErgoScript code generation.
"""

import logging
from typing import Optional

import dspy

from .models import (
    ComplexityLevel,
    ErgoScriptGenerationResult,
    GenerationConfig,
    GenerationContext,
    GenerationResult,
    Query,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


class ContextualGenerate(dspy.Signature):
    """Generate text based on query and retrieved context."""

    query = dspy.InputField(desc="User query or request")
    context = dspy.InputField(desc="Retrieved relevant context from knowledge base")
    answer = dspy.OutputField(desc="Generated response based on context")


class ErgoScriptGenerate(dspy.Signature):
    """Generate ErgoScript code based on requirements and examples."""

    requirements = dspy.InputField(desc="Code requirements and specifications")
    examples = dspy.InputField(desc="Relevant ErgoScript examples from knowledge base")
    documentation = dspy.InputField(desc="Relevant documentation and best practices")
    code = dspy.OutputField(desc="Generated ErgoScript code")
    explanation = dspy.OutputField(desc="Explanation of the generated code")


class CodeReview(dspy.Signature):
    """Review and improve generated ErgoScript code."""

    code = dspy.InputField(desc="ErgoScript code to review")
    requirements = dspy.InputField(desc="Original requirements")
    review = dspy.OutputField(desc="Code review with suggestions for improvement")
    improved_code = dspy.OutputField(desc="Improved version of the code")


class GenerationEngine:
    """
    DSPy-powered generation engine for text and code generation.

    This engine uses DSPy modules to generate responses based on retrieved
    context, with specialized capabilities for ErgoScript code generation.
    """

    def __init__(
        self,
        config: GenerationConfig,
        language_model: Optional[dspy.LM] = None,
    ):
        """
        Initialize the generation engine.

        Args:
            config: Configuration for generation operations
            language_model: DSPy language model to use
        """
        self.config = config

        # Initialize language model
        if language_model:
            self.lm = language_model
        else:
            # Use the new DSPy LM interface for OpenAI models
            model_name = (
                f"openai/{config.model_name}"
                if not config.model_name.startswith("openai/")
                else config.model_name
            )
            self.lm = dspy.LM(
                model=model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop_sequences,
            )

        # Set as default LM for DSPy
        dspy.configure(lm=self.lm)

        # Initialize DSPy modules
        self.contextual_generator = dspy.ChainOfThought(ContextualGenerate)
        self.ergoscript_generator = dspy.ChainOfThought(ErgoScriptGenerate)
        self.code_reviewer = dspy.ChainOfThought(CodeReview)

        logger.info(f"Initialized GenerationEngine with model: {config.model_name}")

    def generate(
        self,
        context: GenerationContext,
        generation_type: str = "general",
    ) -> GenerationResult:
        """
        Generate text based on the provided context.

        Args:
            context: Generation context with query and retrieved documents
            generation_type: Type of generation ("general", "code", "explanation")

        Returns:
            Generation result with generated text and metadata
        """
        logger.debug(
            f"Generating {generation_type} response for query: {context.query.text}"
        )

        if generation_type == "code":
            return self.generate_ergoscript(context)
        elif generation_type == "explanation":
            return self._generate_explanation(context)
        else:
            return self._generate_general(context)

    def generate_ergoscript(
        self,
        context: GenerationContext,
        review_code: bool = True,
    ) -> ErgoScriptGenerationResult:
        """
        Generate ErgoScript code based on requirements and context.

        Args:
            context: Generation context with requirements and examples
            review_code: Whether to review and improve the generated code

        Returns:
            ErgoScript generation result with code and explanation
        """
        logger.debug("Generating ErgoScript code")

        # Prepare context for ErgoScript generation
        requirements = context.query.text
        examples = self._format_code_examples(context.retrieved_documents)
        documentation = self._format_documentation(context.retrieved_documents)

        try:
            # Generate initial code
            result = self.ergoscript_generator(
                requirements=requirements,
                examples=examples,
                documentation=documentation,
            )

            # Safely extract code and explanation from DSPy result
            generated_code = self._extract_text_from_result(result, "code", "")
            explanation = self._extract_text_from_result(result, "explanation", "")

            # Review and improve code if requested
            if review_code and generated_code:
                try:
                    review_result = self.code_reviewer(
                        code=generated_code,
                        requirements=requirements,
                    )

                    # Use improved code if available
                    improved_code = self._extract_text_from_result(
                        review_result, "improved_code", ""
                    )
                    review_text = self._extract_text_from_result(
                        review_result, "review", ""
                    )

                    if improved_code:
                        generated_code = improved_code
                        if review_text:
                            explanation += f"\n\nCode Review: {review_text}"

                except Exception as e:
                    logger.warning(f"Code review failed: {str(e)}")

            # Calculate confidence score based on context quality
            confidence_score = self._calculate_confidence_score(context, generated_code)

            # Extract source document IDs
            source_docs = [doc.document_id for doc in context.retrieved_documents]

            # Estimate complexity
            complexity = self._estimate_code_complexity(generated_code)

            return ErgoScriptGenerationResult(
                generated_text=f"{generated_code}\n\n{explanation}",
                confidence_score=confidence_score,
                reasoning=f"Generated based on {len(context.retrieved_documents)} relevant documents",
                source_documents=source_docs,
                code=generated_code,
                explanation=explanation,
                complexity_estimate=complexity,
                metadata={
                    "generation_type": "ergoscript",
                    "model": self.config.model_name,
                    "temperature": self.config.temperature,
                    "reviewed": review_code,
                    "num_examples": len(
                        [
                            d
                            for d in context.retrieved_documents
                            if d.metadata.category.value == "examples"
                        ]
                    ),
                },
            )

        except Exception as e:
            logger.error(f"ErgoScript generation failed: {str(e)}")

            # Return error result
            return ErgoScriptGenerationResult(
                generated_text=f"Error generating ErgoScript: {str(e)}",
                confidence_score=0.0,
                reasoning="Generation failed due to error",
                source_documents=[],
                code="// Error: Could not generate code",
                explanation=f"An error occurred during code generation: {str(e)}",
                validation_errors=[str(e)],
            )

    def _generate_general(self, context: GenerationContext) -> GenerationResult:
        """Generate general text response."""
        # Format context from retrieved documents
        formatted_context = self._format_context(context.retrieved_documents)

        try:
            result = self.contextual_generator(
                query=context.query.text,
                context=formatted_context,
            )

            generated_text = self._extract_text_from_result(result, "answer", "")
            confidence_score = self._calculate_confidence_score(context, generated_text)
            source_docs = [doc.document_id for doc in context.retrieved_documents]

            return GenerationResult(
                generated_text=generated_text,
                confidence_score=confidence_score,
                reasoning=f"Generated based on {len(context.retrieved_documents)} relevant documents",
                source_documents=source_docs,
                metadata={
                    "generation_type": "general",
                    "model": self.config.model_name,
                    "temperature": self.config.temperature,
                },
            )

        except Exception as e:
            logger.error(f"General generation failed: {str(e)}")

            return GenerationResult(
                generated_text=f"I apologize, but I encountered an error while generating a response: {str(e)}",
                confidence_score=0.0,
                reasoning="Generation failed due to error",
                source_documents=[],
                metadata={"error": str(e)},
            )

    def _generate_explanation(self, context: GenerationContext) -> GenerationResult:
        """Generate explanation-focused response."""
        # Create explanation-focused prompt
        formatted_context = self._format_context(context.retrieved_documents)

        # Modify the query to focus on explanation
        explanation_query = (
            f"Please provide a detailed explanation for: {context.query.text}"
        )

        try:
            result = self.contextual_generator(
                query=explanation_query,
                context=formatted_context,
            )

            generated_text = self._extract_text_from_result(result, "answer", "")
            confidence_score = self._calculate_confidence_score(context, generated_text)
            source_docs = [doc.document_id for doc in context.retrieved_documents]

            return GenerationResult(
                generated_text=generated_text,
                confidence_score=confidence_score,
                reasoning=f"Explanation generated based on {len(context.retrieved_documents)} relevant documents",
                source_documents=source_docs,
                metadata={
                    "generation_type": "explanation",
                    "model": self.config.model_name,
                    "temperature": self.config.temperature,
                },
            )

        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")

            return GenerationResult(
                generated_text=f"I apologize, but I encountered an error while generating an explanation: {str(e)}",
                confidence_score=0.0,
                reasoning="Generation failed due to error",
                source_documents=[],
                metadata={"error": str(e)},
            )

    def _format_context(self, documents: list[RetrievalResult]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant context found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Context {i} (Score: {doc.score:.3f}):")
            if doc.title:
                context_parts.append(f"Title: {doc.title}")
            context_parts.append(doc.content)
            context_parts.append("")  # Empty line separator

        return "\n".join(context_parts)

    def _format_code_examples(self, documents: list[RetrievalResult]) -> str:
        """Format code examples from retrieved documents."""
        examples = [
            doc for doc in documents if doc.metadata.category.value == "examples"
        ]

        if not examples:
            return "No relevant code examples found."

        formatted_examples = []
        for i, doc in enumerate(examples, 1):
            formatted_examples.append(f"Example {i}:")
            if doc.title:
                formatted_examples.append(f"// {doc.title}")
            formatted_examples.append(doc.content)
            formatted_examples.append("")  # Empty line separator

        return "\n".join(formatted_examples)

    def _format_documentation(self, documents: list[RetrievalResult]) -> str:
        """Format documentation from retrieved documents."""
        docs = [
            doc
            for doc in documents
            if doc.metadata.category.value in ["api", "syntax", "best_practices"]
        ]

        if not docs:
            return "No relevant documentation found."

        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            formatted_docs.append(f"Documentation {i}:")
            if doc.title:
                formatted_docs.append(f"Topic: {doc.title}")
            formatted_docs.append(doc.content)
            formatted_docs.append("")  # Empty line separator

        return "\n".join(formatted_docs)

    def _calculate_confidence_score(
        self,
        context: GenerationContext,
        generated_text: str,
    ) -> float:
        """Calculate confidence score for generated text."""
        score = 0.5  # Base score

        # Boost based on number of relevant documents
        num_docs = len(context.retrieved_documents)
        if num_docs >= 5:
            score += 0.2
        elif num_docs >= 3:
            score += 0.1
        elif num_docs == 0:
            score -= 0.3

        # Boost based on average relevance score
        if context.retrieved_documents:
            avg_relevance = (
                sum(doc.score for doc in context.retrieved_documents) / num_docs
            )
            score += avg_relevance * 0.3

        # Boost based on generated text length (reasonable responses)
        text_length = len(generated_text.strip())
        if 100 <= text_length <= 2000:
            score += 0.1
        elif text_length < 50:
            score -= 0.2

        # Ensure score is in valid range
        return max(0.0, min(1.0, score))

    def _extract_text_from_result(
        self, result, attribute: str, default: str = ""
    ) -> str:
        """
        Safely extract text from DSPy result, handling different response types.

        Args:
            result: DSPy result object
            attribute: Attribute name to extract
            default: Default value if extraction fails

        Returns:
            Extracted text as string
        """
        try:
            value = getattr(result, attribute, default)

            # Handle different response types from DSPy
            if isinstance(value, list):
                # DSPy sometimes returns a list of responses
                return value[0] if value else default
            elif hasattr(value, "text"):
                # Sometimes wrapped in an object
                return value.text
            else:
                # Direct string response
                return str(value) if value else default

        except Exception as e:
            logger.warning(f"Failed to extract '{attribute}' from DSPy result: {e}")
            return default

    def _estimate_code_complexity(self, code: str) -> ComplexityLevel:
        """Estimate the complexity of generated ErgoScript code."""
        if not code or code.strip().startswith("//"):
            return ComplexityLevel.BEGINNER

        complexity_indicators = {
            "advanced": ["sigma", "proveDlog", "blake2b", "deserialize", "getVar"],
            "intermediate": ["if", "for", "while", "match", "case", "def"],
            "beginner": ["val", "OUTPUTS", "INPUTS", "HEIGHT"],
        }

        code_lower = code.lower()

        # Count indicators
        advanced_count = sum(
            1
            for indicator in complexity_indicators["advanced"]
            if indicator.lower() in code_lower
        )
        intermediate_count = sum(
            1
            for indicator in complexity_indicators["intermediate"]
            if indicator.lower() in code_lower
        )

        # Determine complexity
        if advanced_count >= 2:
            return ComplexityLevel.ADVANCED
        elif intermediate_count >= 3 or advanced_count >= 1:
            return ComplexityLevel.INTERMEDIATE
        else:
            return ComplexityLevel.BEGINNER


class ErgoScriptGenerator:
    """
    Specialized ErgoScript code generator with validation capabilities.

    This class provides a simplified interface for ErgoScript generation
    with built-in validation and optimization features.
    """

    def __init__(self, generation_engine: GenerationEngine):
        """Initialize with a generation engine."""
        self.generation_engine = generation_engine

    def generate_contract(
        self,
        requirements: str,
        context_documents: list[RetrievalResult],
        complexity: Optional[ComplexityLevel] = None,
    ) -> ErgoScriptGenerationResult:
        """
        Generate a complete ErgoScript contract.

        Args:
            requirements: Contract requirements and specifications
            context_documents: Relevant documents for context
            complexity: Target complexity level

        Returns:
            ErgoScript generation result
        """
        # Create enhanced requirements with complexity guidance
        enhanced_requirements = requirements
        if complexity:
            enhanced_requirements += f"\n\nTarget complexity: {complexity.value}"

        # Create generation context
        query = Query(text=enhanced_requirements)
        context = GenerationContext(
            query=query,
            retrieved_documents=context_documents,
            generation_params={"type": "contract"},
        )

        return self.generation_engine.generate_ergoscript(context)

    def generate_function(
        self,
        function_description: str,
        context_documents: list[RetrievalResult],
    ) -> ErgoScriptGenerationResult:
        """Generate a specific ErgoScript function."""
        enhanced_description = (
            f"Generate an ErgoScript function that: {function_description}"
        )

        query = Query(text=enhanced_description)
        context = GenerationContext(
            query=query,
            retrieved_documents=context_documents,
            generation_params={"type": "function"},
        )

        return self.generation_engine.generate_ergoscript(context)

    def improve_code(
        self,
        existing_code: str,
        improvement_request: str,
        context_documents: list[RetrievalResult],
    ) -> ErgoScriptGenerationResult:
        """Improve existing ErgoScript code."""
        enhanced_request = f"""
        Improve the following ErgoScript code based on this request: {improvement_request}

        Existing code:
        {existing_code}
        """

        query = Query(text=enhanced_request)
        context = GenerationContext(
            query=query,
            retrieved_documents=context_documents,
            generation_params={"type": "improvement", "existing_code": existing_code},
        )

        return self.generation_engine.generate_ergoscript(context)
