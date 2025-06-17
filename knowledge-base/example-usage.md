# Knowledge Base Ingestion - Quick Start Examples

This guide shows practical examples of how to use the new knowledge-base folder structure for easy content ingestion.

## Example 1: Adding Tutorial Documents

Let's say you have some ErgoScript tutorials to add to the knowledge base:

```bash
# 1. Create tutorial documents in the tutorials category
mkdir -p knowledge-base/categories/tutorials
cd knowledge-base/categories/tutorials

# 2. Add your tutorial files (example content)
cat > ergoscript-basics.md << 'EOF'
# ErgoScript Basics Tutorial

This tutorial covers the fundamental concepts of ErgoScript.

## What is ErgoScript?

ErgoScript is a non-Turing complete language for writing smart contracts on the Ergo blockchain...

## Your First Contract

Here's a simple example:
```ergoscript
{
  val threshold = 1000
  sigmaProp(HEIGHT > threshold)
}
```
EOF

# 3. Run ingestion
cd ../../..
python scripts/ingest_knowledge.py --folder categories/tutorials
```

## Example 2: Bulk URL Import

Add multiple documentation URLs for batch processing:

```bash
# 1. Add URLs to the urls.txt file
cat >> knowledge-base/urls.txt << 'EOF'
https://docs.ergoplatform.com/dev/smart-contracts/
https://docs.ergoplatform.com/dev/data-inputs/
https://ergoscript.org/tutorial/
https://github.com/ergoplatform/eips/blob/master/eip-0004.md
EOF

# 2. Run URL ingestion
python scripts/ingest_knowledge.py --verbose

# 3. Check processed URLs
ls -la knowledge-base/processed/urls/
```

## Example 3: GitHub Repository Ingestion

Add entire GitHub repositories to your knowledge base:

```bash
# 1. Add repositories to github-repos.txt
cat >> knowledge-base/github-repos.txt << 'EOF'
ergoplatform/ergoscript-by-example
ergoplatform/eips
scalahub/Kiosk
EOF

# 2. Run repository ingestion
python scripts/ingest_knowledge.py --category examples

# 3. Check ingestion results
python scripts/ingest_knowledge.py --dry-run
```

## Example 4: Mixed Content Ingestion

Process documents, URLs, and repositories together:

```bash
# 1. Organize different types of content
mkdir -p knowledge-base/categories/{tutorials,examples,reference}

# Add some PDF documents
cp ~/Downloads/ergo-whitepaper.pdf knowledge-base/categories/reference/
cp ~/Downloads/ergoscript-advanced.pdf knowledge-base/categories/tutorials/

# Add URLs for scraping
echo "https://docs.ergoplatform.com/mining/" >> knowledge-base/urls.txt

# Add a code repository
echo "ergoplatform/sigma-rust" >> knowledge-base/github-repos.txt

# 2. Run complete ingestion
python scripts/ingest_knowledge.py --force

# 3. Check the results
python -c "
from fintelligence_ai.knowledge import get_knowledge_base_stats
import asyncio
import json
stats = asyncio.run(get_knowledge_base_stats())
print(json.dumps(stats, indent=2))
"
```

## Example 5: Using Custom Configuration

Create a custom configuration for your specific needs:

```bash
# 1. Create custom config
cat > knowledge-base/my-custom-config.json << 'EOF'
{
  "default_category": "defi",
  "default_difficulty": "advanced",
  "file_patterns": {
    "defi": ["*dex*", "*swap*", "*liquidity*", "*yield*"],
    "nft": ["*nft*", "*token*", "*collection*"],
    "mining": ["*mining*", "*pool*", "*hash*"]
  },
  "auto_categorize": true,
  "chunk_size": 1500,
  "chunk_overlap": 300
}
EOF

# 2. Use custom config
python scripts/ingest_knowledge.py --config knowledge-base/my-custom-config.json
```

## Example 6: Monitoring and Maintenance

Track your knowledge base growth and health:

```bash
# 1. Check what's been processed
cat knowledge-base/processed/processed_files.txt

# 2. View detailed logs
tail -f knowledge-base/processed/ingestion.log

# 3. Re-process specific content
python scripts/ingest_knowledge.py --folder categories/examples --force

# 4. Preview what would be processed
python scripts/ingest_knowledge.py --dry-run --verbose

# 5. Get knowledge base statistics
python -c "
import asyncio
import sys
sys.path.insert(0, 'src')
from fintelligence_ai.knowledge import get_knowledge_base_stats

async def show_stats():
    stats = await get_knowledge_base_stats()
    print('Knowledge Base Statistics:')
    for key, value in stats.items():
        print(f'  {key}: {value}')

asyncio.run(show_stats())
"
```

## Example 7: Advanced File Organization

Use filename patterns for automatic categorization:

```bash
# 1. Use descriptive filenames with metadata
mkdir -p knowledge-base/documents

# Add files with category and difficulty in the name
touch knowledge-base/documents/tutorial_beginner_ergoscript-intro.md
touch knowledge-base/documents/example_advanced_multi-sig-contract.es
touch knowledge-base/documents/reference_intermediate_api-guide.pdf

# 2. The ingestion script will automatically categorize these
python scripts/ingest_knowledge.py --verbose

# Files will be categorized as:
# - tutorial_beginner_ergoscript-intro.md -> tutorials (beginner)
# - example_advanced_multi-sig-contract.es -> examples (advanced)
# - reference_intermediate_api-guide.pdf -> reference (intermediate)
```

## Example 8: Error Handling and Recovery

Handle common issues during ingestion:

```bash
# 1. Check for errors in the log
grep -i error knowledge-base/processed/ingestion.log

# 2. Re-run failed ingestions
python scripts/ingest_knowledge.py --force --verbose

# 3. Process only new content (skip already processed)
python scripts/ingest_knowledge.py

# 4. Clean up and restart if needed
rm -rf knowledge-base/processed/
python scripts/ingest_knowledge.py --force
```

## Testing Your Setup

Verify everything is working correctly:

```bash
# 1. Test with a small example
echo "# Test Document" > knowledge-base/documents/test.md
echo "This is a test document for verification." >> knowledge-base/documents/test.md

# 2. Run ingestion
python scripts/ingest_knowledge.py --verbose

# 3. Test RAG retrieval
python -c "
import asyncio
import sys
sys.path.insert(0, 'src')
from fintelligence_ai.knowledge import KnowledgeBaseManager

async def test_search():
    km = KnowledgeBaseManager()
    await km.initialize()
    results = await km.search_knowledge_base('test document', limit=5)
    print('Search Results:')
    for i, result in enumerate(results, 1):
        print(f'{i}. {result}')

asyncio.run(test_search())
"

# 4. Clean up test
rm knowledge-base/documents/test.md
```

## Next Steps

- Check out the [main knowledge base guide](../docs/KNOWLEDGE_BASE.md) for advanced features
- Explore the [API documentation](../docs/API_REFERENCE.md) for programmatic access
- Set up automated ingestion workflows for continuous knowledge updates
- Integrate with your development workflow for documentation-driven development

---

**Happy knowledge building!** 🚀
