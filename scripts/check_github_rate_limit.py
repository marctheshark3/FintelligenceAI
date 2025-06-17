#!/usr/bin/env python3
"""
Quick GitHub Rate Limit Check

This script quickly checks your current GitHub API rate limit status.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Load environment variables
def load_env():
    """Load environment variables from .env file."""
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove inline comments
                    value = value.split("#")[0].strip()
                    os.environ[key] = value


load_env()

from fintelligence_ai.knowledge import GitHubDataCollector


async def main():
    """Check GitHub rate limit status."""
    print("🔍 Checking GitHub API Rate Limit Status")
    print("=" * 40)

    # Check for token
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_API_KEY")
    if token:
        print("✅ GitHub token found in environment")
    else:
        print("⚠️  No GitHub token found - using unauthenticated requests")
        print("   Consider setting GITHUB_TOKEN for higher rate limits")

    # Create collector and check rate limit
    collector = GitHubDataCollector()

    try:
        async with collector:
            rate_status = await collector.check_rate_limit_status()

            if "rate" in rate_status:
                core = rate_status["rate"]
                print("\n📊 Core API Limits:")
                print(f"   Used: {core['used']}")
                print(f"   Remaining: {core['remaining']}")
                print(f"   Total: {core['limit']}")
                print(f"   Reset time: {core['reset']}")

                # Calculate percentage remaining
                percentage = (core["remaining"] / core["limit"]) * 100
                if percentage > 50:
                    print(f"   Status: ✅ Good ({percentage:.1f}% remaining)")
                elif percentage > 10:
                    print(f"   Status: ⚠️  Moderate ({percentage:.1f}% remaining)")
                else:
                    print(f"   Status: 🚨 Low ({percentage:.1f}% remaining)")

                # Recommendations
                print("\n💡 Recommendations:")
                if core["remaining"] < 100:
                    if not token:
                        print(
                            "   - Set up GitHub token: python scripts/setup_github_token.py"
                        )
                        print("   - This increases limit from 60/hour to 5,000/hour")
                    else:
                        print("   - Wait for rate limit reset or try again later")
                else:
                    print("   - You should be able to run knowledge ingestion")

            else:
                print("❌ Could not retrieve rate limit information")
                if "message" in rate_status:
                    print(f"   Error: {rate_status['message']}")

    except Exception as e:
        print(f"❌ Error checking rate limit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
