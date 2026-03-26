import anthropic
import os
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Configuration & Embedding Setup
DB_PATH = "./legal_audit_db"
# Use the Jan 2025 SOTA model (or claude-opus-4-5 if you want max reasoning)
MODEL_NAME = "claude-sonnet-4-5"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

def extract_date_from_source(source_path):
    """
    Extracts YYYY-MM-DD from filenames like '2024-10-12_[EVID]_Email.txt'
    """
    filename = os.path.basename(source_path)
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    return match.group(1) if match else "Unknown Date"

def get_context(query, category=None):
    """Retrieves 10 chunks. Category filtering disabled until re-ingestion."""
    # TODO: Re-ingest data with category metadata to enable filtering
    docs = vectorstore.similarity_search(
        query,
        k=10
        # filter={"category": category}  # Disabled: metadata not set during ingestion
    )

    formatted_context = ""
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        date = extract_date_from_source(source) # ✅ Manual extraction fix
        formatted_context += f"\n[SOURCE: {source} | DATE: {date}]\n{doc.page_content}\n---\n"
    return formatted_context

def run_audit(topic):
    # Fetch truth first, then attorney statements
    truth_data = get_context(topic, "plaintiff_evidence")
    attorney_data = get_context(topic, "attorney_correspondence")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    system_logic = (
        "You are a Senior Legal Malpractice Auditor. You operate at Temperature 0. "
        "Your core directive is to identify when attorney statements deviate from [plaintiff_evidence]. "
        "Strictly adhere to CRPC 1.1, 1.3, and 1.4 rules."
    )

    prompt = f"""
    AUDIT TARGET: {topic}

    [PRIORITY_TRUTH] DATA (Evidence):
    {truth_data}

    ATTORNEY COMMUNICATIONS (Claims):
    {attorney_data}

    ANALYSIS INSTRUCTIONS:
    1. CROSS-REFERENCE: Compare claims to evidence by date.
    2. FLAG: If an attorney denies a fact established in evidence, mark as *** FOUL PLAY DETECTED ***.
    3. GASLIGHTING: Flag any dismissal of evidence as 'emotional' or 'AI-generated'.
    4. LEGALITY: Cite CCP if 'no legal basis' was claimed incorrectly.

    OUTPUT: Markdown Table [Date | Claim | Evidence | Analysis] + Summary.
    """

    response = client.messages.create(
        model=MODEL_NAME, # ✅ Corrected 2025 Model Name
        max_tokens=4096,
        temperature=0,
        system=system_logic,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "employment status discrepancies"
    print(run_audit(topic))
