"""
Validation Agent for FintelligenceAI.

This module implements the ValidationAgent that handles code validation,
syntax checking, semantic analysis, and security assessment for ErgoScript.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

import dspy

from .base import BaseAgent
from .types import (
    AgentCapability,
    AgentConfig,
    AgentRole,
    ConversationContext,
    TaskType,
    ValidationCriteria,
)


class SyntaxValidation(dspy.Signature):
    """Validate ErgoScript syntax."""
    
    code: str = dspy.InputField(desc="ErgoScript code to validate")
    context: str = dspy.InputField(desc="Additional context for validation")
    
    is_valid: bool = dspy.OutputField(desc="Whether syntax is valid")
    errors: str = dspy.OutputField(desc="Syntax errors found")
    suggestions: str = dspy.OutputField(desc="Suggestions for fixing errors")


class SemanticAnalysis(dspy.Signature):
    """Perform semantic analysis of ErgoScript code."""
    
    code: str = dspy.InputField(desc="ErgoScript code to analyze")
    use_case: str = dspy.InputField(desc="Use case context")
    
    is_semantically_correct: bool = dspy.OutputField(desc="Whether semantics are correct")
    issues: str = dspy.OutputField(desc="Semantic issues found")
    recommendations: str = dspy.OutputField(desc="Recommendations for improvement")


class SecurityAnalysis(dspy.Signature):
    """Analyze ErgoScript code for security vulnerabilities."""
    
    code: str = dspy.InputField(desc="ErgoScript code to analyze")
    context: str = dspy.InputField(desc="Security context and requirements")
    
    security_score: float = dspy.OutputField(desc="Security score (0-10)")
    vulnerabilities: str = dspy.OutputField(desc="Security vulnerabilities found")
    mitigations: str = dspy.OutputField(desc="Recommended security mitigations")


class ValidationAgent(BaseAgent):
    """
    Validation Agent specialized in ErgoScript code validation.
    
    This agent handles syntax validation, semantic analysis, security assessment,
    and compilation testing for generated ErgoScript code.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Validation Agent."""
        if config is None:
            config = self._create_default_config()
        
        super().__init__(config)
        self._initialize_dspy_modules()
        
        # Validation rules and patterns
        self.syntax_patterns = self._load_syntax_patterns()
        self.security_patterns = self._load_security_patterns()
        
        self.logger.info("Validation Agent initialized")
    
    def _create_default_config(self) -> AgentConfig:
        """Create default configuration for Validation Agent."""
        capabilities = [
            AgentCapability(
                name="syntax_validation",
                description="Validate ErgoScript syntax and structure",
                supported_task_types=[TaskType.SYNTAX_CHECK, TaskType.CODE_VALIDATION],
                required_tools=["syntax_analyzer"]
            )
        ]
        
        return AgentConfig(
            agent_id="validation_agent_001",
            role=AgentRole.VALIDATION,
            name="Validation Agent",
            description="Specialized agent for ErgoScript code validation",
            capabilities=capabilities,
            max_concurrent_tasks=3,
            timeout_seconds=180
        )
    
    def _initialize_dspy_modules(self) -> None:
        """Initialize DSPy modules for validation tasks."""
        self.dspy_modules.update({
            "syntax_validator": dspy.ChainOfThought(SyntaxValidation),
        })
    
    def _load_syntax_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load syntax validation patterns."""
        return {
            "curly_braces": {
                "pattern": r"^\s*\{.*\}\s*$",
                "description": "ErgoScript must be wrapped in curly braces",
                "severity": "error"
            },
            "sigma_prop": {
                "pattern": r"sigmaProp\s*\(",
                "description": "ErgoScript should end with sigmaProp(...)",
                "severity": "warning"
            },
            "val_declaration": {
                "pattern": r"val\s+\w+\s*=",
                "description": "Variable declarations should use 'val'",
                "severity": "info"
            },
            "box_access": {
                "pattern": r"(SELF|INPUTS|OUTPUTS)\s*\(",
                "description": "Box access patterns",
                "severity": "info"
            }
        }
    
    def _load_security_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load security analysis patterns."""
        return {
            "unchecked_value": {
                "pattern": r"\.value\s*[><=!]",
                "description": "Direct value comparisons without bounds checking",
                "severity": "high",
                "mitigation": "Add bounds checking for value comparisons"
            },
            "unsafe_token_access": {
                "pattern": r"\.tokens\(\d+\)\._[12]",
                "description": "Direct token access without existence check",
                "severity": "medium",
                "mitigation": "Check token existence before accessing properties"
            },
            "missing_guard": {
                "pattern": r"sigmaProp\(true\)",
                "description": "Always-true condition creates security risk",
                "severity": "critical",
                "mitigation": "Replace with proper guard conditions"
            },
            "register_access": {
                "pattern": r"\.R\d+\[.*\]\.get",
                "description": "Direct register access without safe extraction",
                "severity": "medium",
                "mitigation": "Use getOrElse or isDefined for safe register access"
            }
        }
    
    async def _execute_task_impl(
        self,
        task_type: TaskType,
        content: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute validation-specific tasks."""
        if task_type == TaskType.SYNTAX_CHECK:
            return await self._handle_syntax_validation(content, context, metadata)
        elif task_type == TaskType.CODE_VALIDATION:
            return await self._handle_code_validation(content, context, metadata)
        else:
            raise ValueError(f"Unsupported task type for Validation Agent: {task_type}")
    
    async def _handle_syntax_validation(
        self,
        code: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle syntax validation requests."""
        self.logger.info("Performing syntax validation")
        
        try:
            validator = self.dspy_modules["syntax_validator"]
            result = validator(
                code=code,
                context="ErgoScript syntax validation"
            )
            
            return {
                "is_valid": result.is_valid,
                "syntax_errors": result.errors,
                "suggestions": result.suggestions,
                "metadata": {
                    "validation_type": "syntax"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in syntax validation: {e}")
            raise
    
    async def _handle_code_validation(
        self,
        code: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle comprehensive code validation."""
        # For now, just do syntax validation
        return await self._handle_syntax_validation(code, context, metadata)
    
    async def _handle_semantic_analysis(
        self,
        code: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle semantic analysis requests."""
        self.logger.info("Performing semantic analysis")
        
        try:
            # Determine use case
            use_case = "general"
            if metadata and "use_case" in metadata:
                use_case = metadata["use_case"]
            elif context and context.context_data:
                use_case = context.context_data.get("use_case", "general")
            
            # DSPy-based semantic analysis
            analyzer = self.dspy_modules["semantic_analyzer"]
            result = analyzer(
                code=code,
                use_case=use_case
            )
            
            # Rule-based semantic checks
            semantic_issues = self._check_semantic_rules(code, use_case)
            
            return {
                "is_semantically_correct": result.is_semantically_correct,
                "semantic_issues": result.issues,
                "rule_based_issues": semantic_issues,
                "recommendations": result.recommendations,
                "use_case": use_case,
                "metadata": {
                    "analysis_type": "semantic",
                    "use_case_analyzed": use_case,
                    "ai_analysis": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in semantic analysis: {e}")
            raise
    
    async def _handle_comprehensive_validation(
        self,
        code: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle comprehensive validation requests."""
        self.logger.info("Performing comprehensive validation")
        
        try:
            # Get validation criteria
            criteria = ValidationCriteria()
            if metadata and "validation_criteria" in metadata:
                criteria = ValidationCriteria(**metadata["validation_criteria"])
            
            # Perform all validations in parallel
            validation_tasks = []
            
            if criteria.syntax_check:
                validation_tasks.append(self._handle_syntax_validation(code, context, metadata))
            
            if criteria.semantic_check:
                validation_tasks.append(self._handle_semantic_analysis(code, context, metadata))
            
            if criteria.security_check:
                validation_tasks.append(self._handle_security_analysis(code, context, metadata))
            
            # Wait for all validations to complete
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Compile comprehensive results
            comprehensive_result = {
                "overall_valid": True,
                "validation_results": {},
                "summary": {
                    "total_issues": 0,
                    "critical_issues": 0,
                    "warnings": 0
                },
                "recommendations": [],
                "metadata": {
                    "validation_criteria": criteria.dict(),
                    "validations_performed": len(validation_tasks)
                }
            }
            
            # Process results
            if criteria.syntax_check and len(results) > 0 and not isinstance(results[0], Exception):
                syntax_result = results[0]
                comprehensive_result["validation_results"]["syntax"] = syntax_result
                if not syntax_result["is_valid"]:
                    comprehensive_result["overall_valid"] = False
                    comprehensive_result["summary"]["total_issues"] += 1
            
            if criteria.semantic_check and len(results) > 1 and not isinstance(results[1], Exception):
                semantic_result = results[1]
                comprehensive_result["validation_results"]["semantic"] = semantic_result
                if not semantic_result["is_semantically_correct"]:
                    comprehensive_result["overall_valid"] = False
                    comprehensive_result["summary"]["total_issues"] += 1
            
            if criteria.security_check and len(results) > 2 and not isinstance(results[2], Exception):
                security_result = results[2]
                comprehensive_result["validation_results"]["security"] = security_result
                if security_result["security_score"] < 7.0:
                    comprehensive_result["overall_valid"] = False
                    comprehensive_result["summary"]["critical_issues"] += 1
            
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive validation: {e}")
            raise
    
    async def _handle_security_analysis(
        self,
        code: str,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle security analysis requests."""
        self.logger.info("Performing security analysis")
        
        try:
            # Rule-based security checks
            security_issues = self._check_security_patterns(code)
            
            # DSPy-based security analysis
            analyzer = self.dspy_modules["security_analyzer"]
            security_context = self._build_security_context(context, metadata)
            
            result = analyzer(
                code=code,
                context=security_context
            )
            
            return {
                "security_score": result.security_score,
                "vulnerabilities": result.vulnerabilities,
                "rule_based_issues": security_issues,
                "mitigations": result.mitigations,
                "metadata": {
                    "analysis_type": "security",
                    "patterns_checked": len(self.security_patterns),
                    "ai_analysis": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in security analysis: {e}")
            raise
    
    def _check_syntax_rules(self, code: str) -> List[Dict[str, Any]]:
        """Check code against syntax rules."""
        issues = []
        
        for rule_name, rule_config in self.syntax_patterns.items():
            pattern = rule_config["pattern"]
            
            if rule_name == "curly_braces":
                if not re.search(pattern, code, re.DOTALL):
                    issues.append({
                        "rule": rule_name,
                        "severity": rule_config["severity"],
                        "description": rule_config["description"],
                        "location": "global"
                    })
            elif rule_name == "sigma_prop":
                if not re.search(pattern, code):
                    issues.append({
                        "rule": rule_name,
                        "severity": rule_config["severity"],
                        "description": "Consider ending with sigmaProp(...)",
                        "location": "end"
                    })
        
        return issues
    
    def _check_semantic_rules(self, code: str, use_case: str) -> List[Dict[str, Any]]:
        """Check code against semantic rules."""
        issues = []
        
        # Use case specific checks
        if use_case == "token":
            if "tokens(" not in code:
                issues.append({
                    "rule": "token_usage",
                    "severity": "warning",
                    "description": "Token contracts should reference tokens",
                    "suggestion": "Add token validation logic"
                })
        
        elif use_case == "auction":
            if "value" not in code:
                issues.append({
                    "rule": "auction_value",
                    "severity": "warning",
                    "description": "Auction contracts should handle value comparisons",
                    "suggestion": "Add bid value validation"
                })
        
        return issues
    
    def _check_security_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Check code against security patterns."""
        issues = []
        
        for pattern_name, pattern_config in self.security_patterns.items():
            matches = re.finditer(pattern_config["pattern"], code)
            
            for match in matches:
                issues.append({
                    "pattern": pattern_name,
                    "severity": pattern_config["severity"],
                    "description": pattern_config["description"],
                    "mitigation": pattern_config["mitigation"],
                    "location": {
                        "start": match.start(),
                        "end": match.end(),
                        "matched_text": match.group()
                    }
                })
        
        return issues
    
    def _build_validation_context(
        self,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build context string for validation."""
        context_parts = []
        
        if context and context.context_data:
            context_parts.append(f"Domain: {context.context_data.get('domain', 'general')}")
            context_parts.append(f"Use case: {context.context_data.get('use_case', 'general')}")
        
        if metadata:
            if "use_case" in metadata:
                context_parts.append(f"Use case: {metadata['use_case']}")
            if "complexity_level" in metadata:
                context_parts.append(f"Complexity: {metadata['complexity_level']}")
        
        return "; ".join(context_parts) if context_parts else "General validation"
    
    def _build_security_context(
        self,
        context: Optional[ConversationContext] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build context string for security analysis."""
        context_parts = ["ErgoScript security analysis"]
        
        if metadata and "security_requirements" in metadata:
            context_parts.append(f"Requirements: {metadata['security_requirements']}")
        
        return "; ".join(context_parts) 