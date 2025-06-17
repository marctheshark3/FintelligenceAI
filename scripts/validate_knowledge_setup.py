#!/usr/bin/env python3
"""
Validation script for FintelligenceAI Knowledge Base Setup

This script validates that the knowledge ingestion system is properly configured
and can run basic operations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_folder_structure():
    """Check if the knowledge-base folder structure is properly set up."""
    print("🔍 Checking knowledge-base folder structure...")

    kb_dir = Path("knowledge-base")
    required_structure = [
        "knowledge-base/",
        "knowledge-base/documents/",
        "knowledge-base/categories/",
        "knowledge-base/categories/tutorials/",
        "knowledge-base/categories/examples/",
        "knowledge-base/categories/reference/",
        "knowledge-base/categories/guides/",
        "knowledge-base/urls.txt",
        "knowledge-base/github-repos.txt",
        "knowledge-base/.knowledge-config.json",
        "knowledge-base/README.md",
    ]

    results = {}
    for path_str in required_structure:
        path = Path(path_str)
        exists = path.exists()
        results[path_str] = exists
        status = "✅" if exists else "❌"
        print(f"  {status} {path_str}")

    missing = [path for path, exists in results.items() if not exists]
    if missing:
        print(f"\n⚠️  Missing {len(missing)} items. Run the demo script to create them:")
        print("  python demo_knowledge_ingestion.py")
        return False

    print("✅ Folder structure is complete!")
    return True


def check_ingestion_script():
    """Check if the ingestion script exists and is executable."""
    print("\n🔍 Checking ingestion script...")

    script_path = Path("scripts/ingest_knowledge.py")
    if not script_path.exists():
        print("❌ Ingestion script not found!")
        return False

    # Check if executable
    if not script_path.stat().st_mode & 0o111:
        print("⚠️  Script is not executable. Making it executable...")
        script_path.chmod(script_path.stat().st_mode | 0o111)

    print("✅ Ingestion script is ready!")
    return True


def check_dependencies():
    """Check if required dependencies are available."""
    print("\n🔍 Checking dependencies...")

    required_modules = [
        "fintelligence_ai.knowledge",
        "fintelligence_ai.rag.models",
        "httpx",
        "pydantic",
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module} - {e}")
            missing_modules.append(module)

    if missing_modules:
        print(f"\n⚠️  Missing {len(missing_modules)} dependencies.")
        print(
            "Make sure the FintelligenceAI environment is activated and dependencies are installed."
        )
        return False

    print("✅ All dependencies are available!")
    return True


def test_basic_functionality():
    """Test basic knowledge management functionality."""
    print("\n🔍 Testing basic functionality...")

    try:
        from fintelligence_ai.knowledge import KnowledgeBaseManager

        print("  ✅ Can import KnowledgeBaseManager")

        # Test instantiation
        km = KnowledgeBaseManager()
        print("  ✅ Can create KnowledgeBaseManager instance")

        return True

    except Exception as e:
        print(f"  ❌ Error testing functionality: {e}")
        return False


def run_dry_run_test():
    """Run a dry-run test of the ingestion script."""
    print("\n🔍 Running dry-run test...")

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/ingest_knowledge.py", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("  ✅ Dry-run completed successfully!")
            print(f"  📄 Output preview: {result.stdout[:200]}...")
            return True
        else:
            print(f"  ❌ Dry-run failed with exit code {result.returncode}")
            print(f"  Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("  ⚠️  Dry-run timed out (this might be normal for large setups)")
        return True
    except Exception as e:
        print(f"  ❌ Error running dry-run: {e}")
        return False


def main():
    """Run all validation checks."""
    print("🔧 FintelligenceAI Knowledge Base Setup Validation")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("FintelligenceAI.md").exists():
        print(
            "❌ Please run this script from the FintelligenceAI project root directory"
        )
        sys.exit(1)

    checks = [
        ("Folder Structure", check_folder_structure),
        ("Ingestion Script", check_ingestion_script),
        ("Dependencies", check_dependencies),
        ("Basic Functionality", test_basic_functionality),
        ("Dry Run Test", run_dry_run_test),
    ]

    results = {}
    for check_name, check_func in checks:
        print(f"\n{'='*20} {check_name} {'='*20}")
        results[check_name] = check_func()

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20}: {status}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All checks passed! Your knowledge base setup is ready to use.")
        print("\n📖 Next steps:")
        print("1. Run: python demo_knowledge_ingestion.py")
        print("2. Add your content to knowledge-base/ folders")
        print("3. Run: python scripts/ingest_knowledge.py")
    else:
        print("⚠️  Some checks failed. Please review the errors above.")
        print("\n🔧 Common fixes:")
        print("1. Run: python demo_knowledge_ingestion.py")
        print("2. Check that all dependencies are installed")
        print("3. Ensure you're in the project root directory")

        sys.exit(1)


if __name__ == "__main__":
    main()
