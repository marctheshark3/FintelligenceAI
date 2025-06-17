# FintelligenceAI Knowledge Base Ingestion

This folder provides an easy way to add new content to the FintelligenceAI knowledge base. Simply place your files, URLs, or GitHub links in the appropriate folders and run the ingestion script.

## 📁 Folder Structure

```
knowledge-base/
├── documents/           # Place files here (PDF, MD, TXT, etc.)
├── urls.txt            # List of URLs to scrape (one per line)
├── github-repos.txt    # List of GitHub repositories (one per line)
├── categories/         # Organize documents by category
│   ├── tutorials/
│   ├── examples/
│   ├── reference/
│   └── guides/
└── processed/          # Auto-created - contains processed files
```

## 🚀 Quick Start

### Option 1: Local Embeddings (No API Keys Required)

1. **Set up Ollama**: Run `python scripts/setup_ollama_embeddings.py`
2. **Add Documents**: Drop files into `documents/` or appropriate category folder
3. **Add URLs**: Add URLs to `urls.txt` (one per line)
4. **Add GitHub Repos**: Add repository URLs to `github-repos.txt`
5. **Run Ingestion**: Execute `python scripts/ingest_knowledge.py`

### Option 2: OpenAI Embeddings

1. **Set API Key**: Add `OPENAI_API_KEY=your_key` to `.env` file
2. **Disable Local Mode**: Set `DSPY_LOCAL_MODE=false` in `.env`
3. **Follow steps 2-5** from Option 1 above

## 📄 Supported File Types

- **Documents**: PDF, Markdown (.md), Text (.txt), Word (.docx)
- **Code**: Python (.py), ErgoScript (.es), JavaScript (.js)
- **Data**: JSON (.json), CSV (.csv)
- **Web**: HTML files

## 🏷️ File Organization

### Categories

Organize files by placing them in category subfolders:

- `categories/tutorials/` - Learning materials and guides
- `categories/examples/` - Code examples and implementations
- `categories/reference/` - API docs and technical references
- `categories/guides/` - Best practices and how-to guides

### Metadata via Filename

Include metadata in filenames using this pattern:
```
[category]_[difficulty]_filename.ext
```

Examples:
- `tutorial_beginner_ergoscript-basics.md`
- `example_advanced_multi-signature.es`
- `reference_intermediate_api-docs.pdf`

## 🌐 URL Ingestion

Add URLs to `urls.txt`, one per line:

```
https://docs.ergoplatform.com/dev/smart-contracts/
https://github.com/ergoplatform/eips/blob/master/eip-0004.md
https://ergoscript.org/tutorial/
```

Supported sites:
- Documentation sites (Sphinx, GitBook, etc.)
- GitHub pages and wikis
- Blog posts and articles
- Knowledge bases

## 🐙 GitHub Repository Ingestion

Add repository URLs to `github-repos.txt`:

```
https://github.com/ergoplatform/ergoscript-examples
https://github.com/ergoplatform/eips
ergoplatform/sigma-rust
```

The script will:
- Clone or fetch repository content
- Extract markdown files, documentation, and code
- Process according to repository structure

## ⚙️ Advanced Configuration

Create a `.knowledge-config.json` file to customize ingestion:

```json
{
  "default_category": "general",
  "default_difficulty": "intermediate",
  "file_patterns": {
    "tutorials": ["*tutorial*", "*guide*", "*howto*"],
    "examples": ["*example*", "*demo*", "*sample*"],
    "reference": ["*api*", "*ref*", "*spec*"]
  },
  "auto_categorize": true,
  "extract_metadata": true,
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

## 🔄 Running Ingestion

### Enhanced Ingestion Commands

#### Basic Ingestion
```bash
# Ingest all new content with real-time progress tracking
python scripts/ingest_knowledge.py

# Ingest specific folder
python scripts/ingest_knowledge.py --folder categories/tutorials

# Ingest with custom category
python scripts/ingest_knowledge.py --category advanced_topics

# Preview changes without processing
python scripts/ingest_knowledge.py --dry-run

# Force re-processing of all files
python scripts/ingest_knowledge.py --force --verbose
```

#### Content Visualization & Management
```bash
# Show hierarchical tree of processed content
python scripts/ingest_knowledge.py --show-tree

# Display detailed GitHub repository manifest
python scripts/ingest_knowledge.py --show-manifest

# Clear entire vector database and start fresh
python scripts/ingest_knowledge.py --clear-db

# Remove specific repository from tracking
python scripts/ingest_knowledge.py --remove-repo "https://github.com/microsoft/vscode-docs"
```

### All Available Options
- `--dry-run` - Preview what would be ingested
- `--force` - Re-ingest already processed files
- `--category` - Override default category
- `--verbose` - Show detailed progress
- `--config` - Use custom config file
- `--show-tree` - Display content tree and exit
- `--show-manifest` - Display GitHub manifest and exit
- `--clear-db` - Clear vector database (with confirmation)
- `--remove-repo URL` - Remove specific repository from tracking

## 📊 Enhanced Progress Monitoring

### Real-Time Progress Tracking
The enhanced ingestion system provides comprehensive progress visualization:

```bash
🐙 Cloning & processing GitHub repos: [███████████████░░░░░░░░░░░░░░░]   50.0% (1/2) | ETA: 44s | Current: Processing eips files
```

**Features**:
- **Visual Progress Bars**: Unicode-based progress visualization
- **ETA Calculation**: Real-time estimated time to completion
- **Processing Speed**: Items/second and time per item metrics
- **Current Status**: Shows exactly what's being processed
- **Phase Tracking**: Separate progress for cloning vs. processing

### Content Visualization
```bash
# View processed content tree
python scripts/ingest_knowledge.py --show-tree

# Example output:
🌳 Processed Content Tree:
────────────────────────────────────────
📊 Summary:
   🐙 GitHub Repositories: 2
   📄 Local Files: 1
   📝 File Types: .md(23), .py(6)
   📂 Categories: reference(2), examples(5), general(22)

📁 GitHub Repositories
├── 🐙 ergo-python-appkit
│   ├── 📄 📊 6 files processed
│   └── 📄 📂 Examples: 5 files
└── 🐙 eips
    └── 📄 📊 23 files processed
```

### GitHub Repository Details
```bash
# View detailed repository manifest
python scripts/ingest_knowledge.py --show-manifest

# Shows specific file names, categories, and processing timestamps
```

### Traditional Monitoring
The script will also:
- Log detailed progress for each file/URL/repo
- Track any errors or warnings
- Create comprehensive summary reports
- Maintain processed files in `processed/` folder
- Generate detailed logs in `processed/ingestion.log`

## 🛠️ Troubleshooting

### Common Issues

**Files not being processed:**
- Check file permissions
- Ensure supported file format
- Verify file is not corrupted

**URL scraping fails:**
- Check internet connection
- Verify URL is accessible
- Some sites may block automated access

**GitHub processing issues:**
- **Performance**: New git clone method is ~100x faster than API calls
- **No rate limiting**: Local cloning bypasses GitHub API limits
- **Git availability**: Ensure git is installed: `git --version`
- **Network access**: Verify GitHub accessibility: `curl -I https://github.com`
- **Legacy rate limiting**: If using old API method, check: `python scripts/check_github_rate_limit.py`

**Database management issues:**
```bash
# Clear everything and start fresh
python scripts/ingest_knowledge.py --clear-db

# Remove problematic repository
python scripts/ingest_knowledge.py --remove-repo "https://github.com/problematic/repo"

# Check current state
python scripts/ingest_knowledge.py --show-tree
python scripts/ingest_knowledge.py --show-manifest
```

**Performance issues:**
```bash
# Check what will be processed
python scripts/ingest_knowledge.py --dry-run --verbose

# Monitor disk space during processing
df -h

# Process incrementally
python scripts/ingest_knowledge.py --folder categories/small_category
```

### Getting Help

Check multiple sources for detailed error information:
- Real-time logs: `tail -f knowledge-base/processed/ingestion.log`
- Error analysis: `grep -i error knowledge-base/processed/ingestion.log`
- Processing stats: `grep "processed repository" knowledge-base/processed/ingestion.log`
- Content overview: `python scripts/ingest_knowledge.py --show-tree`

## 📝 Examples

### Example 1: Adding Tutorial Documents
```bash
# 1. Copy tutorial PDFs to categories/tutorials/
cp ~/Downloads/ergoscript-tutorial.pdf knowledge-base/categories/tutorials/

# 2. Run ingestion
python scripts/ingest_knowledge.py --category tutorials

# 3. Check results
tail -f knowledge-base/processed/ingestion.log
```

### Example 2: Bulk URL Import
```bash
# 1. Create URL list
echo "https://docs.ergoplatform.com/dev/" >> knowledge-base/urls.txt
echo "https://ergoscript.org/docs/" >> knowledge-base/urls.txt

# 2. Run ingestion
python scripts/ingest_knowledge.py --verbose

# 3. Review processed content
ls knowledge-base/processed/urls/
```

### Example 3: Enhanced GitHub Repository Processing
```bash
# 1. Add repository
echo "https://github.com/ergoplatform/eips" >> knowledge-base/github-repos.txt

# 2. Run ingestion with real-time progress
python scripts/ingest_knowledge.py --category examples

# 3. View processed content tree
python scripts/ingest_knowledge.py --show-tree

# 4. Check detailed repository manifest
python scripts/ingest_knowledge.py --show-manifest

# 5. Verify specific files processed
# Output shows exact file names, categories, and processing details
```

### Example 4: Database Management Workflow
```bash
# 1. Check current state
python scripts/ingest_knowledge.py --show-tree

# 2. Remove unwanted repository
python scripts/ingest_knowledge.py --remove-repo "https://github.com/microsoft/vscode-docs"

# 3. Clear everything if needed
python scripts/ingest_knowledge.py --clear-db

# 4. Re-process with selected content
echo "https://github.com/ergoplatform/eips" > knowledge-base/github-repos.txt
python scripts/ingest_knowledge.py

# 5. Verify final state
python scripts/ingest_knowledge.py --show-manifest
```

### Example 5: Performance Monitoring
```bash
# 1. Preview processing load
python scripts/ingest_knowledge.py --dry-run --verbose

# 2. Monitor real-time during processing
python scripts/ingest_knowledge.py &
tail -f knowledge-base/processed/ingestion.log

# 3. Analyze performance after completion
grep "Duration:" knowledge-base/processed/ingestion.log
grep "Processing speed:" knowledge-base/processed/ingestion.log
```

---

**Need more help?** Check the main documentation at [docs/KNOWLEDGE_BASE.md](../docs/KNOWLEDGE_BASE.md)
