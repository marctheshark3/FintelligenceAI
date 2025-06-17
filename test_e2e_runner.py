#!/usr/bin/env python3
"""
E2E Provider Testing Runner

Simple script to run the comprehensive end-to-end tests for all AI providers.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


async def main():
    """Run the E2E provider tests."""
    print("🚀 Starting E2E Provider Testing...")

    try:
        # Import the test module
        from tests.integration.test_e2e_providers import (
            AVAILABLE_PROVIDERS,
            TestE2EProviders,
        )

        # Show available providers
        print("\n📋 Available Providers:")
        for provider in AVAILABLE_PROVIDERS:
            status = "✅ Available" if provider.is_available() else "❌ Not Available"
            print(f"  • {provider.name}: {status}")
            if not provider.is_available():
                missing_vars = [
                    var for var in provider.required_env_vars if not os.getenv(var)
                ]
                if missing_vars:
                    print(f"    Missing env vars: {missing_vars}")

        # Create test instance
        test_instance = TestE2EProviders()

        # Run the main E2E test
        await test_instance.test_all_providers_e2e()

        print("\n🎉 All tests completed successfully!")

    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
