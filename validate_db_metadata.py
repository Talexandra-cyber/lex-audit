"""
Legal Audit DB - Provenance & Integrity Validator (v2025.12)
"""
import os
import re
from collections import Counter
from langchain_community.embeddings import HuggingFaceEmbeddings # Fixed import for your venv
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
DB_PATH = "./legal_audit_db"
EXPECTED_CATEGORIES = ["plaintiff_evidence", "attorney_correspondence"]

print("="*80)
print(f"AUDIT DATABASE VALIDATION: {os.path.abspath(DB_PATH)}")
print("="*80)

# Initialize
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

total_chunks = vectorstore._collection.count()
print(f"Total Chunks Found: {total_chunks}")

if total_chunks == 0:
    print("❌ ERROR: Database is empty. Verify persist_directory.")
    exit()

# Sample Documents
results = vectorstore.similarity_search("evidence correspondence deposition", k=50)

# Counters and Data Collection
cat_counter = Counter()
tag_counter = Counter()
date_status = {"valid": 0, "missing": 0}

print("\n1. CATEGORY FIELD ANALYSIS (Internal Metadata)")
for doc in results:
    cat = doc.metadata.get('category', 'MISSING')
    cat_counter[cat] += 1

    # Extract tags/dates from filename (Source)
    source = doc.metadata.get('source', 'Unknown')
    tags = re.findall(r'\[([A-Z]+)\]', source)
    tag_counter.update(tags)

    if re.search(r'\d{4}-\d{2}-\d{2}', source):
        date_status["valid"] += 1
    else:
        date_status["missing"] += 1

for cat, count in cat_counter.items():
    status = "✅" if cat in EXPECTED_CATEGORIES else "⚠️ UNEXPECTED"
    print(f"   - {cat}: {count} chunks {status}")

print("\n2. FILENAME TAGS (Provenance Check)")
for tag, count in tag_counter.most_common():
    print(f"   - [{tag}]: {count} occurrences in sources")

print(f"\n3. DATE EXTRACTION POTENTIAL")
print(f"   - Documents with YYYY-MM-DD in source: {date_status['valid']}/50")
if date_status["missing"] > 0:
    print(f"   - ⚠️ WARNING: {date_status['missing']} docs missing dates in filename.")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
