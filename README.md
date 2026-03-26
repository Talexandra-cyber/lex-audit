# Lex — Forensic Document Intelligence Engine

A retrieval-augmented generation system for forensic legal analysis. Lex ingests large document collections, builds a categorized vector database, and uses Claude to cross-reference evidence against claims — surfacing inconsistencies, timeline violations, and compliance gaps with citation-level precision.

## What It Does

Legal review at scale is a manual, error-prone process. Lex automates the hardest part: finding where the record contradicts the narrative.

Given a corpus of legal documents (correspondence, filings, depositions, evidence), Lex:

1. Ingests and chunks all documents into a categorized ChromaDB vector store
2. Retrieves contextually relevant chunks per query across multiple collections
3. Sends structured context to Claude at **Temperature 0** for deterministic, auditable analysis
4. Outputs a citation-grounded markdown report flagging discrepancies, timeline violations, and rule-of-professional-conduct deviations

## Scale

| Metric | Value |
|---|---|
| Documents indexed | 214 PDFs |
| Total chunks | 7,801 |
| Embedding model | `all-MiniLM-L6-v2` (local, no API cost) |
| LLM | Claude Sonnet (Anthropic API) |
| Vector store | ChromaDB (local persistence) |

## Architecture

```
PDF Corpus
    │
    ▼
[Ingestion Pipeline]
  ├─ PyPDFLoader — parallel PDF extraction
  ├─ RecursiveCharacterTextSplitter — 1000 token chunks, 100 overlap
  ├─ Category tagging — plaintiff_evidence / attorney_correspondence / discovery / billing / etc.
  └─ HuggingFace embeddings → ChromaDB (persisted locally)
    │
    ▼
[ForensicRAGEngine]
  ├─ Multi-collection retrieval (category-filtered similarity search)
  ├─ Date-sorted context assembly
  └─ Source attribution per chunk
    │
    ▼
[Claude Sonnet — Temperature 0]
  ├─ Cross-reference: evidence vs. claims
  ├─ Timeline analysis: chronological consistency
  ├─ CRPC compliance check (Rules 1.1, 1.3, 1.4)
  └─ Output: Markdown table [Date | Claim | Evidence | Analysis | Verdict]
```

## Modules

| File | Purpose |
|---|---|
| `audit_engine.py` | Core audit runner — retrieves dual-collection context, runs Claude analysis |
| `forensic_audit_v5.py` | Line-by-line forensic judgment engine — applies CRPC rules to every communication |
| `ingest_server.py` | Document ingestion entry point — auto-detects document directory |
| `ingest_case_file.py` | Category-aware ingestion with folder-to-metadata mapping |
| `validate_db_metadata.py` | DB integrity validator — checks chunk counts, category distribution, date coverage |
| `benchmark_parallel.py` | Parallel ingestion benchmarking — tests 1/2/4/8 worker configurations |

## Document Category Schema

The ingestion pipeline maps folder structure to semantic categories:

| Folder | Category | Subcategory |
|---|---|---|
| Medical Records | `evidence` | `medical_records` |
| Deposition | `discovery` | `deposition` |
| Motions | `pleadings` | `motions` |
| Orders-Rulings | `court_filings` | `orders` |
| Invoices-Billing | `billing` | `invoices` |
| Settlement Docs | `settlement` | `settlement_docs` |
| Subpoenaed Docs | `evidence` | `subpoenaed` |

## Setup

```bash
git clone https://github.com/Talexandra-cyber/lex-audit
cd lex-audit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

Add your documents to `./documents/` (organized in subfolders by category), then ingest:
```bash
python ingest_server.py
```

Run an audit:
```bash
python audit_engine.py
```

Validate your database:
```bash
python validate_db_metadata.py
```

## Design Decisions

**Temperature 0** — Legal analysis must be deterministic and reproducible. No creativity, no hallucinated citations.

**Local embeddings** — `all-MiniLM-L6-v2` runs entirely on-device. Documents never leave the machine during ingestion — only query-time context reaches the Claude API.

**Dual-collection retrieval** — Evidence and correspondence are retrieved separately, then presented to Claude as labeled context blocks. This prevents the model from conflating source types.

**Date extraction from filenames** — Documents named with ISO dates (`YYYY-MM-DD_doctype.pdf`) are automatically sorted chronologically in context, preserving timeline integrity.

**Citation-grounded output** — Every finding includes `[SOURCE: filename | DATE: YYYY-MM-DD]` so every claim in the analysis can be traced back to a specific document chunk.
