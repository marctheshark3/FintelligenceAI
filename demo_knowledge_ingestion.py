#!/usr/bin/env python3
"""
Demo script for FintelligenceAI Knowledge Base Ingestion

This script demonstrates how to use the new knowledge-base folder structure
for easy content ingestion.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fintelligence_ai.knowledge import get_knowledge_base_stats


def setup_demo_content():
    """Set up demo content in the knowledge-base folder."""
    print("🔧 Setting up demo content...")

    # Create knowledge-base structure if it doesn't exist
    kb_dir = Path("knowledge-base")
    kb_dir.mkdir(exist_ok=True)

    # Create demo tutorial
    tutorials_dir = kb_dir / "categories" / "tutorials"
    tutorials_dir.mkdir(parents=True, exist_ok=True)

    demo_tutorial = tutorials_dir / "demo_ergoscript_tutorial.md"
    if not demo_tutorial.exists():
        demo_tutorial.write_text(
            """# ErgoScript Demo Tutorial

This is a demo tutorial for testing the knowledge base ingestion system.

## What is ErgoScript?

ErgoScript is a powerful smart contract language for the Ergo blockchain.
It enables secure and efficient decentralized applications.

## Basic Example

Here's a simple ErgoScript contract:

```ergoscript
{
  val threshold = 1000
  sigmaProp(HEIGHT > threshold)
}
```

This contract allows spending only after a certain blockchain height.

## Key Features

- Non-Turing complete for security
- Cryptographic operations built-in
- UTXO model support
- Zero-knowledge proof integration

## Next Steps

1. Learn about box model
2. Understand sigma propositions
3. Practice with examples
4. Build your first dApp
"""
        )
        print(f"✅ Created demo tutorial: {demo_tutorial}")

    # Add demo URLs
    urls_file = kb_dir / "urls.txt"
    demo_urls = [
        "# Demo URLs for testing",
        "# https://docs.ergoplatform.com/dev/smart-contracts/",
        "# https://ergoscript.org/tutorial/",
    ]
    if not urls_file.exists():
        urls_file.write_text("\n".join(demo_urls))
        print(f"✅ Created demo URLs file: {urls_file}")

    # Add demo GitHub repos
    repos_file = kb_dir / "github-repos.txt"
    demo_repos = [
        "# Demo GitHub repositories",
        "# ergoplatform/ergoscript-by-example",
        "# Note: Uncomment lines above to actually ingest",
    ]
    if not repos_file.exists():
        repos_file.write_text("\n".join(demo_repos))
        print(f"✅ Created demo repos file: {repos_file}")

    # Create example document
    docs_dir = kb_dir / "documents"
    docs_dir.mkdir(exist_ok=True)

    demo_doc = docs_dir / "example_advanced_nft-contract.md"
    if not demo_doc.exists():
        demo_doc.write_text(
            """# Advanced NFT Contract Example

This document demonstrates an advanced NFT contract implementation.

## Overview

This contract implements a sophisticated NFT with the following features:
- Royalty payments
- Transfer restrictions
- Metadata updates
- Collection management

## Contract Code

```ergoscript
{
  val royaltyRate = 250 // 2.5%
  val creator = OUTPUTS(0).R4[Coll[Byte]].get
  val isValidTransfer = {
    val royaltyBox = OUTPUTS(1)
    val royaltyAmount = OUTPUTS(0).value * royaltyRate / 10000
    royaltyBox.value >= royaltyAmount &&
    royaltyBox.propositionBytes == creator
  }

  sigmaProp(isValidTransfer)
}
```

## Security Considerations

1. Always validate royalty payments
2. Check creator signatures
3. Verify metadata integrity
4. Test transfer scenarios

This is classified as an advanced example for experienced developers.
"""
        )
        print(f"✅ Created demo document: {demo_doc}")


def show_folder_structure():
    """Display the knowledge-base folder structure."""
    print("\n📁 Knowledge Base Folder Structure:")
    kb_dir = Path("knowledge-base")

    if not kb_dir.exists():
        print("❌ knowledge-base folder not found!")
        return

    def print_tree(path, prefix=""):
        """Print a tree structure of the directory."""
        items = sorted(path.iterdir())
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")

            if item.is_dir() and not item.name.startswith("."):
                next_prefix = prefix + ("    " if is_last else "│   ")
                try:
                    print_tree(item, next_prefix)
                except PermissionError:
                    pass

    print_tree(kb_dir)


async def run_demo_ingestion():
    """Run the demo ingestion process."""
    print("\n🚀 Running demo ingestion...")

    # Import here to avoid issues if the system isn't set up
    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/ingest_knowledge.py", "--dry-run", "--verbose"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        print("📋 Ingestion Script Output:")
        print("=" * 50)
        print(result.stdout)
        if result.stderr:
            print("\n⚠️  Warnings/Errors:")
            print(result.stderr)
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error running ingestion script: {e}")
        print("💡 Try running manually: python scripts/ingest_knowledge.py --dry-run")


async def show_knowledge_base_stats():
    """Show current knowledge base statistics."""
    print("\n📊 Knowledge Base Statistics:")
    try:
        stats = await get_knowledge_base_stats()
        print("=" * 30)
        for key, value in stats.items():
            print(f"{key:20}: {value}")
        print("=" * 30)
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        print("💡 Make sure the knowledge base is initialized")


def show_usage_examples():
    """Show practical usage examples."""
    print("\n💡 Usage Examples:")
    print("=" * 50)

    examples = [
        (
            "Add a tutorial document:",
            "cp my-tutorial.md knowledge-base/categories/tutorials/",
        ),
        (
            "Add URLs to scrape:",
            "echo 'https://docs.ergoplatform.com/dev/' >> knowledge-base/urls.txt",
        ),
        (
            "Add GitHub repository:",
            "echo 'ergoplatform/eips' >> knowledge-base/github-repos.txt",
        ),
        ("Run ingestion:", "python scripts/ingest_knowledge.py"),
        (
            "Ingest specific folder:",
            "python scripts/ingest_knowledge.py --folder categories/tutorials",
        ),
        ("Force re-ingestion:", "python scripts/ingest_knowledge.py --force"),
        ("Preview changes:", "python scripts/ingest_knowledge.py --dry-run"),
    ]

    for description, command in examples:
        print(f"\n{description}")
        print(f"  $ {command}")

    print("\n📚 For more examples, see: knowledge-base/example-usage.md")


async def main():
    """Main demo function."""
    print("🤖 FintelligenceAI Knowledge Base Ingestion Demo")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("FintelligenceAI.md").exists():
        print("❌ Please run this demo from the FintelligenceAI project root directory")
        return

    # Set up demo content
    setup_demo_content()

    # Show folder structure
    show_folder_structure()

    # Run demo ingestion (dry run)
    await run_demo_ingestion()

    # Show current stats
    await show_knowledge_base_stats()

    # Show usage examples
    show_usage_examples()

    print("\n🎉 Demo complete!")
    print("\n📖 Next steps:")
    print("1. Add your own documents to knowledge-base/documents/")
    print("2. Add URLs to knowledge-base/urls.txt")
    print("3. Add GitHub repos to knowledge-base/github-repos.txt")
    print("4. Run: python scripts/ingest_knowledge.py")
    print("5. Check knowledge-base/README.md for detailed instructions")


if __name__ == "__main__":
    asyncio.run(main())
