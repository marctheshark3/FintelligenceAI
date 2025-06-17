#!/usr/bin/env python3
"""
GitHub Token Setup Script

This script helps set up GitHub authentication for the knowledge ingestion
process to avoid rate limits.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio

import httpx


async def check_github_rate_limit(token: str = None) -> dict:
    """Check current GitHub API rate limit status."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "FintelligenceAI-Setup/1.0",
    }

    if token:
        headers["Authorization"] = f"token {token}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.github.com/rate_limit", headers=headers
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_env_file_path() -> Path:
    """Get the path to the .env file."""
    project_root = Path(__file__).parent.parent
    return project_root / ".env"


def load_existing_token() -> str:
    """Load existing GitHub token from environment."""
    return os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_API_KEY") or ""


def save_token_to_env(token: str):
    """Save the GitHub token to .env file."""
    env_file = get_env_file_path()

    if env_file.exists():
        # Read existing content
        with open(env_file) as f:
            content = f.read()

        # Check if GITHUB_TOKEN already exists
        if "GITHUB_TOKEN=" in content:
            # Replace existing token
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("GITHUB_TOKEN="):
                    lines[i] = f"GITHUB_TOKEN={token}"
                    break
            content = "\n".join(lines)
        else:
            # Add new token
            content += f"\n# GitHub API Token\nGITHUB_TOKEN={token}\n"

        with open(env_file, "w") as f:
            f.write(content)
    else:
        # Create new .env file
        with open(env_file, "w") as f:
            f.write(f"# GitHub API Token\nGITHUB_TOKEN={token}\n")

    print(f"✅ GitHub token saved to {env_file}")


async def validate_token(token: str) -> bool:
    """Validate that the GitHub token works."""
    try:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {token}",
            "User-Agent": "FintelligenceAI-Setup/1.0",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.github.com/user", headers=headers)
            if response.status_code == 200:
                user_data = response.json()
                print(
                    f"✅ Token validated for user: {user_data.get('login', 'Unknown')}"
                )
                return True
            else:
                print(
                    f"❌ Token validation failed: {response.status_code} - {response.text}"
                )
                return False
    except Exception as e:
        print(f"❌ Token validation error: {e}")
        return False


async def main():
    """Main setup function."""
    print("🔧 GitHub Token Setup for FintelligenceAI")
    print("=" * 50)

    # Check current rate limit
    print("\n📊 Checking current rate limit status...")
    existing_token = load_existing_token()
    rate_limit = await check_github_rate_limit(existing_token)

    if "error" in rate_limit:
        print(f"❌ Error checking rate limit: {rate_limit['error']}")
    else:
        core_limit = rate_limit.get("rate", {})
        remaining = core_limit.get("remaining", 0)
        limit = core_limit.get("limit", 0)

        if existing_token:
            print(f"✅ Authenticated requests: {remaining}/{limit} remaining")
            if remaining > 100:
                print("✅ Current rate limit is sufficient for knowledge ingestion")
                return
        else:
            print(f"⚠️  Unauthenticated requests: {remaining}/{limit} remaining")
            print("⚠️  This is very low for knowledge ingestion!")

    # Prompt for token setup
    print("\n🔑 GitHub Token Setup")
    print(
        "To increase rate limits from 60/hour to 5,000/hour, you need a GitHub Personal Access Token."
    )
    print("\nSteps to create a token:")
    print("1. Go to https://github.com/settings/tokens")
    print("2. Click 'Generate new token (classic)'")
    print("3. Give it a descriptive name like 'FintelligenceAI Knowledge Ingestion'")
    print("4. Select scope: 'public_repo' (for public repositories)")
    print("5. Click 'Generate token'")
    print("6. Copy the token (you won't see it again!)")

    choice = (
        input("\nDo you want to set up a GitHub token now? (y/n): ").lower().strip()
    )

    if choice in ["y", "yes"]:
        token = input("\nPaste your GitHub Personal Access Token: ").strip()

        if not token:
            print("❌ No token provided. Exiting.")
            return

        # Validate token
        print("\n🔍 Validating token...")
        if await validate_token(token):
            # Save token
            save_token_to_env(token)

            # Check new rate limit
            print("\n📊 Checking new rate limit...")
            new_rate_limit = await check_github_rate_limit(token)
            if "error" not in new_rate_limit:
                core_limit = new_rate_limit.get("rate", {})
                remaining = core_limit.get("remaining", 0)
                limit = core_limit.get("limit", 0)
                print(f"✅ New rate limit: {remaining}/{limit} requests remaining")

            print(
                "\n🎉 Setup complete! You can now run knowledge ingestion without rate limits."
            )
            print("Run: python scripts/ingest_knowledge.py --force")
        else:
            print("❌ Token validation failed. Please check your token and try again.")
    else:
        print("\n⚠️  Continuing without GitHub token. Rate limits may cause issues.")
        print(
            "You can run this script again later: python scripts/setup_github_token.py"
        )


if __name__ == "__main__":
    asyncio.run(main())
