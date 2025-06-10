#!/usr/bin/env python3
"""
System Validation Script for FintelligenceAI
Validates all components and capabilities are working correctly.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SystemValidator:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        self.results = []

    async def validate_health_checks(self) -> dict:
        """Validate all service health endpoints"""
        logger.info("🔍 Running health checks...")

        services = {
            "API": f"{self.base_url}/health",
            "API_Agents": f"{self.base_url}/agents/health",
            "ChromaDB": "http://localhost:8100/api/v1/heartbeat",
        }

        results = {}
        for service, endpoint in services.items():
            try:
                logger.info(f"  Checking {service}...")
                response = await self.client.get(endpoint)
                results[service] = {
                    "status": "✅ HEALTHY"
                    if response.status_code == 200
                    else f"❌ UNHEALTHY ({response.status_code})",
                    "response_time": f"{response.elapsed.total_seconds():.2f}s",
                    "details": response.json()
                    if response.status_code == 200
                    else response.text[:200],
                }
                logger.info(f"    {service}: {results[service]['status']}")
            except Exception as e:
                results[service] = {"status": "❌ ERROR", "error": str(e)}
                logger.error(f"    {service}: {results[service]['status']} - {str(e)}")

        return results

    async def validate_agent_capabilities(self) -> dict:
        """Test all agent capabilities"""
        logger.info("🤖 Validating agent capabilities...")

        tests = [
            {
                "name": "System Status",
                "endpoint": "/agents/status",
                "method": "GET",
                "payload": None,
            },
            {
                "name": "Research Agent",
                "endpoint": "/agents/research",
                "method": "POST",
                "payload": {"query": "What is ErgoScript?", "max_results": 3},
            },
            {
                "name": "Simple Code Generation",
                "endpoint": "/agents/generate-code/simple",
                "method": "POST",
                "payload": {
                    "requirements": "Create a simple token contract",
                    "complexity": "beginner",
                },
            },
            {
                "name": "Code Validation",
                "endpoint": "/agents/validate-code",
                "method": "POST",
                "payload": {
                    "code": "val myBox = SELF",
                    "context": "Simple ErgoScript validation",
                },
            },
        ]

        results = {}
        for test in tests:
            try:
                logger.info(f"  Testing {test['name']}...")
                if test["method"] == "GET":
                    response = await self.client.get(
                        f"{self.base_url}{test['endpoint']}"
                    )
                else:
                    response = await self.client.post(
                        f"{self.base_url}{test['endpoint']}", json=test["payload"]
                    )

                results[test["name"]] = {
                    "status": "✅ PASSED"
                    if response.status_code == 200
                    else f"❌ FAILED ({response.status_code})",
                    "response_time": f"{response.elapsed.total_seconds():.2f}s",
                    "response": response.json()
                    if response.status_code == 200
                    else response.text[:200],
                }
                logger.info(f"    {test['name']}: {results[test['name']]['status']}")
            except Exception as e:
                results[test["name"]] = {"status": "❌ ERROR", "error": str(e)}
                logger.error(f"    {test['name']}: ERROR - {str(e)}")

        return results

    async def validate_optimization_features(self) -> dict:
        """Test DSPy optimization capabilities"""
        logger.info("⚡ Validating optimization features...")

        tests = [
            {
                "name": "Optimization Status",
                "endpoint": "/optimization/metrics/summary",
                "method": "GET",
            },
            {
                "name": "Quick Evaluation",
                "endpoint": "/optimization/evaluate",
                "method": "POST",
                "payload": {"component": "rag_pipeline", "quick_eval": True},
            },
        ]

        results = {}
        for test in tests:
            try:
                logger.info(f"  Testing {test['name']}...")
                if test["method"] == "GET":
                    response = await self.client.get(
                        f"{self.base_url}{test['endpoint']}"
                    )
                else:
                    response = await self.client.post(
                        f"{self.base_url}{test['endpoint']}", json=test["payload"]
                    )

                results[test["name"]] = {
                    "status": "✅ PASSED"
                    if response.status_code == 200
                    else f"❌ FAILED ({response.status_code})",
                    "response_time": f"{response.elapsed.total_seconds():.2f}s",
                }
                logger.info(f"    {test['name']}: {results[test['name']]['status']}")
            except Exception as e:
                results[test["name"]] = {"status": "❌ ERROR", "error": str(e)}
                logger.error(f"    {test['name']}: ERROR - {str(e)}")

        return results

    async def validate_full_workflow(self) -> dict:
        """Test a complete end-to-end workflow"""
        logger.info("🔄 Testing complete workflow...")

        try:
            # Step 1: Research
            logger.info("  Step 1: Research query...")
            research_response = await self.client.post(
                f"{self.base_url}/agents/research",
                json={
                    "query": "How to create a basic ErgoScript contract?",
                    "max_results": 2,
                },
            )

            if research_response.status_code != 200:
                return {
                    "status": "❌ FAILED",
                    "step": "research",
                    "error": research_response.text,
                }

            # Step 2: Generate code
            logger.info("  Step 2: Code generation...")
            generation_response = await self.client.post(
                f"{self.base_url}/agents/generate-code/simple",
                json={
                    "requirements": "Create a simple box that stores a value",
                    "complexity": "beginner",
                },
            )

            if generation_response.status_code != 200:
                return {
                    "status": "❌ FAILED",
                    "step": "generation",
                    "error": generation_response.text,
                }

            # Step 3: Validate generated code
            try:
                generated_code = generation_response.json().get(
                    "generated_code", "val placeholder = 1"
                )
            except:
                generated_code = "val placeholder = 1"

            logger.info("  Step 3: Code validation...")
            validation_response = await self.client.post(
                f"{self.base_url}/agents/validate-code",
                json={"code": generated_code, "context": "Generated code validation"},
            )

            return {
                "status": "✅ PASSED"
                if validation_response.status_code == 200
                else "❌ FAILED",
                "steps_completed": 3,
                "total_time": f"{(research_response.elapsed + generation_response.elapsed + validation_response.elapsed).total_seconds():.2f}s",
            }

        except Exception as e:
            return {"status": "❌ ERROR", "error": str(e)}

    async def run_full_validation(self) -> dict:
        """Run complete system validation"""
        logger.info("🚀 Starting full system validation...")
        start_time = time.time()

        validation_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "validation_duration": "0s",
            "health_checks": await self.validate_health_checks(),
            "agent_capabilities": await self.validate_agent_capabilities(),
            "optimization": await self.validate_optimization_features(),
            "end_to_end_workflow": await self.validate_full_workflow(),
        }

        # Calculate overall status
        all_passed = True
        total_tests = 0
        passed_tests = 0

        for category, results in validation_results.items():
            if category in ["timestamp", "validation_duration"]:
                continue
            if isinstance(results, dict):
                for test, result in results.items():
                    total_tests += 1
                    if isinstance(result, dict) and "✅" in result.get("status", ""):
                        passed_tests += 1
                    elif "❌" in str(result.get("status", "")):
                        all_passed = False

        validation_results["overall_status"] = (
            "✅ ALL TESTS PASSED"
            if all_passed
            else f"❌ {passed_tests}/{total_tests} TESTS PASSED"
        )
        validation_results["validation_duration"] = f"{time.time() - start_time:.2f}s"

        return validation_results

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()


async def main():
    """Main validation function"""
    print("🔍 FINTELLIGENCE AI SYSTEM VALIDATION")
    print("=" * 60)

    async with SystemValidator() as validator:
        try:
            results = await validator.run_full_validation()

            # Save results
            results_file = Path("validation_results.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            # Print summary
            print("\n" + "=" * 60)
            print("📊 VALIDATION SUMMARY")
            print("=" * 60)
            print(f"Overall Status: {results['overall_status']}")
            print(f"Validation Duration: {results['validation_duration']}")
            print(f"Report saved to: {results_file.absolute()}")

            # Print detailed results
            print("\n📋 DETAILED RESULTS:")
            for category, category_results in results.items():
                if category in ["timestamp", "validation_duration", "overall_status"]:
                    continue
                print(f"\n{category.replace('_', ' ').title()}:")
                if isinstance(category_results, dict):
                    for test, result in category_results.items():
                        if isinstance(result, dict):
                            status = result.get("status", "Unknown")
                            print(f"  • {test}: {status}")
                        else:
                            print(f"  • {test}: {result}")

            print("=" * 60)

            # Exit with appropriate code
            if "❌" in results["overall_status"]:
                sys.exit(1)
            else:
                print("🎉 All validations passed successfully!")
                sys.exit(0)

        except KeyboardInterrupt:
            print("\n⚠️ Validation interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Validation failed with error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
