#!/usr/bin/env python3
"""
Forensic Legal Audit System v5 - Line-by-Line Email Judgment Engine
Applies CRPC rules to every attorney-client communication
"""
import os
import sys
import re
import anthropic
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_PATH = "./legal_audit_db"
MODEL = "claude-sonnet-4-5"

class ForensicRAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)

    def retrieve(self, query_topic, top_k=10, category=None):
        search_filter = {"category": category} if category else None
        docs = self.vectorstore.similarity_search(query_topic, k=top_k, filter=search_filter)
        docs.sort(key=lambda x: x.metadata.get('date', '9999-99-99'))
        
        context = ""
        for doc in docs:
            date = doc.metadata.get('date', 'Unknown')
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            context += f"[{date}] (Source: {source}):\n{doc.page_content}\n\n---\n\n"
        return context

def run_line_by_line_audit(query_topic, rag_engine, client):
    print(f"\n{'='*80}")
    print(f"LINE-BY-LINE FORENSIC JUDGMENT ENGINE")
    print(f"Topic: {query_topic}")
    print(f"{'='*80}\n")

    print("Retrieving ALL ATTORNEY_CORRESPONDENCE for line-by-line analysis...")
    all_correspondence = rag_engine.retrieve(query_topic, top_k=50, category="attorney_correspondence")

    print("Retrieving PRIORITY_TRUTH for cross-reference...")
    truth_context = rag_engine.retrieve(query_topic, top_k=20, category="priority_truth")

    print("Retrieving PLAINTIFF_EVIDENCE...")
    evidence_context = rag_engine.retrieve(query_topic, top_k=15, category="plaintiff_evidence")

    full_context = f"""
[ALL ATTORNEY_CORRESPONDENCE - Line-by-Line Analysis]:
{all_correspondence}

[PRIORITY_TRUTH - Cross-Reference]:
{truth_context}

[PLAINTIFF_EVIDENCE]:
{evidence_context}
"""

    # LINE-BY-LINE FORENSIC JUDGMENT PROMPT
    prompt = f"""[ROLE]: Forensic Judgement Engine - Line-by-Line Email Auditor applying California Rules of Professional Conduct to every communication.

[TASK]: Analyze EVERY email/communication in [ATTORNEY_CORRESPONDENCE] and apply the 5 forensic judgment categories below.

[CONTEXT]:
{full_context}

[FORENSIC JUDGMENT CATEGORIES]:

**CATEGORY 1: PSYCHOLOGICAL ABUSE & GASLIGHTING**
- **Trigger Phrases:** "AI program", "AI generated", "AI fiction", "metamorphosis", "merchandising", "emotional", "confused", "untethered from reality"
- **Rule:** CRPC 3.3 (Candor) / CRPC 8.4(c) (Dishonesty)
- **Logic:** IF attorney denies reality of client's career/health history documented in sworn testimony → FLAG "BAD FAITH DISPARAGEMENT"
- **Cross-reference:** October Deposition Page 16 (DoD Contractor status)
- **Action:** Create table entry showing date of disparagement + days since deposition attendance

**CATEGORY 2: WILLFUL BLINDNESS (DISCOVERY ABANDONMENT)**
- **Trigger Phrases:** "typically do not order", "additional costs", "not worth pursuing", "can't justify expense"
- **Rule:** CRPC 1.1 (Competence) / CRPC 1.3 (Diligence)
- **Logic:** IF attorney admits refusing to obtain evidence to save firm costs WHILE collecting 40-45% contingency fee → FLAG "MATERIAL BREACH"
- **Context:** Fee structure = $120k-$135k potential on $300k policy
- **Action:** Calculate cost-benefit ratio (avoided costs vs. potential fee)

**CATEGORY 3: FINANCIAL EXTORTION / FIDUCIARY BREACH**
- **Trigger Phrases:** "FedEx", "billing", "your account", "personal expense", "shipping costs"
- **Rule:** CRPC 1.8(e)(1) - Paragraph 8 (Attorney must advance costs)
- **Logic:** IF firm extracted $11,213+ in fees/costs BUT forces client to pay for check delivery (~$20-50) → FLAG "PARAGRAPH 8 LAPSE" + "PETTY EXPLOITATION"
- **Context:** Settlement = $8,000 to client after fees deducted
- **Action:** Calculate extraction ratio (fees extracted ÷ client recovery)

**CATEGORY 4: TACTICAL SUPPRESSION (THE BIG STALL)**
- **Trigger Phrases:** "IME", "demand", "forwarding", "received", "notification"
- **Rule:** CRPC 1.3 (Diligence) / CRPC 1.4 (Communication)
- **Logic:** IF email shows IME demand received on Date A BUT client not notified until Date B → Calculate delay days
  - IF delay > 10 days AND Date B is <20 days before major holiday → FLAG "TACTICAL SABOTAGE"
  - Calculate: CCP §2032.310 window = 16 days notice + holiday closure = impossible timeline
- **Target:** Nov 25 IME received → Dec 8 client notified = 13-day delay
- **Action:** Create timeline showing: Received date → Notification date → Holiday period → Exam date → Days available vs. Days required

**CATEGORY 5: INEPT LITIGATION (THE 16-MONTH GAP)**
- **Trigger Phrases:** "rejected", "clerical", "missing signature", "no order", "typo", "OSC", "sanctions", "failure to file"
- **Rule:** CRPC 1.1 (Competence) / CCP §575.2 (Sanctions)
- **Logic:** IF email documents filing rejection/error → FLAG "PROFESSIONAL INCOMPETENCE"
  - IF client billed for rejection/re-filing fees → FLAG "UNJUST ENRICHMENT"
  - IF OSC issued for sanctions → FLAG "SYSTEMIC INCOMPETENCE"
- **Timeline:** Mar 2, 2023 FAC permission → Jul 12, 2024 FAC filed = 497 days
- **Action:** Count total filing errors, sum error-related fees charged to client, identify OSC dates

[ANALYSIS INSTRUCTIONS]:

For EACH email in [ATTORNEY_CORRESPONDENCE]:
1. **Extract metadata:** Date, sender, subject line
2. **Scan body text** for trigger phrases (case-insensitive)
3. **Apply judgment category** when trigger detected
4. **Cross-reference** with PRIORITY_TRUTH to verify disparagement
5. **Calculate temporal data:** Days between events, delay periods
6. **Quantify financial impact:** Fees, costs, extraction ratios

[OUTPUT FORMAT]:

Generate a **Markdown table** with these columns:
| Date | Email Subject/Context | Trigger Category | Specific Phrase | CRPC Violation | Quantified Harm | Cross-Reference |

Followed by:
1. **Category Summary:** Count of violations per category
2. **Financial Extraction Analysis:** Total fees vs. client recovery
3. **Temporal Analysis:** Delay calculations (IME notification, filing delays)
4. **Pattern Evidence:** Systemic vs. isolated incidents

[CRITICAL REQUIREMENTS]:
- Analyze EVERY communication in chronological order
- Flag ALL trigger phrases (do not summarize or skip)
- Calculate exact day counts for delays
- Cross-reference disparagement claims with deposition dates
- Show mathematical precision (no estimates)

Temperature: 0. Line-by-line precision. No summarization.
"""

    print("Executing Line-by-Line Forensic Judgment with Claude Sonnet 4.5...\n")

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0,
        system="You are a forensic email auditor. Analyze every communication for CRPC violations. Temperature 0. Line-by-line precision.",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text

def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    if len(sys.argv) > 1:
        target_topic = " ".join(sys.argv[1:])
    else:
        target_topic = "line by line email audit gaslighting IME delay FedEx costs"

    try:
        rag_engine = ForensicRAGEngine()
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        report = run_line_by_line_audit(target_topic, rag_engine, client)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = target_topic.replace(' ', '_')[:40]
    filename = f"line_by_line_audit_{safe_topic}_{timestamp}.md"

    with open(filename, "w") as f:
        f.write(f"# LINE-BY-LINE FORENSIC JUDGMENT REPORT\n")
        f.write(f"## Email-by-Email CRPC Violation Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Analysis Model:** Claude Sonnet 4.5 (Temperature 0)\n")
        f.write(f"**Methodology:** Line-by-line trigger detection across all attorney-client communications\n")
        f.write(f"**Audit Scope:** {target_topic}\n\n")
        f.write("---\n\n")
        f.write(report)

    print(f"\n{'='*80}")
    print(f"✅ Line-by-Line Audit Complete. Saved to: {filename}")
    print(f"{'='*80}\n")
    print(report)

if __name__ == "__main__":
    main()
