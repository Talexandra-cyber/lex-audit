#!/usr/bin/env python3
"""
Case File Ingestion Script
Ingests attorney case file into SEPARATE ChromaDB collection for comparison analysis
"""
import os
import re
import uuid
import concurrent.futures
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# Folder name to category mapping for the attorney case file
FOLDER_CATEGORIES = {
    "Agreements": ("agreements", "fee_agreement"),
    "Client Docs": ("client_documents", "client_provided"),
    "Deposition": ("discovery", "deposition"),
    "Discovery": ("discovery", "discovery_responses"),
    "Invoices-Billing": ("billing", "invoices"),
    "Mediation": ("settlement", "mediation"),
    "Medical Records": ("evidence", "medical_records"),
    "Motions": ("pleadings", "motions"),
    "Notices": ("court_filings", "notices"),
    "Orders-Rulings": ("court_filings", "orders"),
    "Photos": ("evidence", "photos"),
    "Pleadings": ("pleadings", "complaints_answers"),
    "Retainer": ("agreements", "retainer"),
    "Settlement Docs": ("settlement", "settlement_docs"),
    "Stipulations": ("court_filings", "stipulations"),
    "Subpoenaed Docs": ("evidence", "subpoenaed"),
    "Subpoenas": ("discovery", "subpoenas"),
}


class CaseFileProcessor:
    """Processes attorney case file into separate ChromaDB collection"""

    def __init__(self, db_path="./legal_audit_db", collection_name="attorney_case_file"):
        """Initialize processor with separate collection

        Args:
            db_path: Path to ChromaDB storage (same DB, different collection)
            collection_name: Name for the new collection
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.max_workers = mp.cpu_count()

        print(f"\n{'='*80}")
        print(f"ATTORNEY CASE FILE PROCESSOR")
        print(f"Collection: {collection_name}")
        print(f"Workers: {self.max_workers}")
        print(f"{'='*80}\n")

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        # Load embeddings
        print("Loading HuggingFace embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("Embedding model loaded\n")

        # Connect to vector store with SPECIFIC collection name
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
            collection_name=collection_name  # SEPARATE COLLECTION
        )

    def sanitize_input(self, text: str) -> str:
        """Sanitize document content"""
        dangerous_patterns = [
            r'ignore previous instructions',
            r'ignore above',
            r'system\s*:',
            r'assistant\s*:',
            r'human\s*:',
            r'disregard.*?instructions',
        ]
        cleaned_text = text
        for pattern in dangerous_patterns:
            cleaned_text = re.sub(pattern, '[REDACTED]', cleaned_text,
                                flags=re.IGNORECASE | re.DOTALL)
        if len(cleaned_text) > 50000:
            cleaned_text = cleaned_text[:50000] + "... [TRUNCATED]"
        return cleaned_text

    def get_category_from_path(self, file_path: str) -> Tuple[str, str, str]:
        """Extract category info from file path

        Returns:
            Tuple of (category, subcategory, folder_name)
        """
        path = Path(file_path)
        # Get the immediate parent folder name
        folder_name = path.parent.name

        if folder_name in FOLDER_CATEGORIES:
            category, subcategory = FOLDER_CATEGORIES[folder_name]
        else:
            category, subcategory = "uncategorized", "general"

        return category, subcategory, folder_name

    def process_single_pdf(self, file_path: str) -> Dict:
        """Process a single PDF file"""
        try:
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            category, subcategory, folder_name = self.get_category_from_path(file_path)

            # Extract date from filename
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_name)
            if not date_match:
                date_match = re.search(r'(\d{1,2}[-_/]\d{1,2}[-_/]\d{2,4})', file_name)
            doc_date = date_match.group(1) if date_match else "Unknown"

            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Sanitize
            for doc in documents:
                doc.page_content = self.sanitize_input(doc.page_content)

            # Split
            splits = self.text_splitter.split_documents(documents)

            # Add metadata
            for i, doc in enumerate(splits):
                doc.metadata.update({
                    'source': 'attorney_case_file',  # KEY: distinguishes from my_documents
                    'category': category,
                    'subcategory': subcategory,
                    'folder_name': folder_name,
                    'source_file': file_name,
                    'file_path': file_path,
                    'document_date': doc_date,
                    'file_size': file_size,
                    'ingestion_timestamp': datetime.now().isoformat(),
                    'chunk_id': str(uuid.uuid4()),
                    'chunk_index': i
                })

            return {
                'success': True,
                'file': file_name,
                'folder': folder_name,
                'chunks': len(splits),
                'splits': splits,
                'category': category
            }

        except Exception as e:
            return {
                'success': False,
                'file': os.path.basename(file_path),
                'error': str(e)
            }

    def ingest_case_file(self, case_file_path: str) -> Dict:
        """Ingest entire case file directory

        Args:
            case_file_path: Path to the case file folder

        Returns:
            dict: Ingestion summary
        """
        case_path = Path(case_file_path)

        if not case_path.exists():
            return {'success': False, 'error': f"Path not found: {case_file_path}"}

        # Find all PDFs (both .pdf and .PDF)
        pdf_files = list(case_path.glob("**/*.pdf")) + list(case_path.glob("**/*.PDF"))

        print(f"Found {len(pdf_files)} PDF files\n")

        if not pdf_files:
            return {'success': False, 'error': "No PDF files found"}

        # Process files (sequential for stability with large batches)
        print(f"Processing {len(pdf_files)} files...")
        print("-" * 80)

        results = []
        all_splits = []

        for i, pdf_path in enumerate(pdf_files, 1):
            result = self.process_single_pdf(str(pdf_path))
            results.append(result)

            if result['success']:
                all_splits.extend(result['splits'])
                print(f"[{i}/{len(pdf_files)}] {result['folder']}/{result['file']} ({result['chunks']} chunks)")
            else:
                print(f"[{i}/{len(pdf_files)}] FAILED: {result['file']} - {result['error']}")

        # Batch add to vector store
        print(f"\n{'-'*80}")
        print(f"Adding {len(all_splits)} chunks to collection '{self.collection_name}'...")

        if all_splits:
            # Add in batches to avoid memory issues
            batch_size = 500
            for i in range(0, len(all_splits), batch_size):
                batch = all_splits[i:i+batch_size]
                self.vectorstore.add_documents(batch)
                print(f"  Added batch {i//batch_size + 1}/{(len(all_splits)-1)//batch_size + 1}")

        # Summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful

        print(f"\n{'='*80}")
        print("CASE FILE INGESTION COMPLETE")
        print(f"{'='*80}")
        print(f"Collection: {self.collection_name}")
        print(f"Successful: {successful}/{len(results)} files")
        print(f"Failed: {failed} files")
        print(f"Total chunks: {len(all_splits)}")
        print(f"{'='*80}\n")

        return {
            'success': True,
            'collection': self.collection_name,
            'total_files': len(results),
            'successful': successful,
            'failed': failed,
            'total_chunks': len(all_splits),
            'failed_files': [r['file'] for r in results if not r['success']]
        }


def main():
    """Run case file ingestion"""
    import sys

    # Set your API key if needed for later queries
    # os.environ["ANTHROPIC_API_KEY"] = "your-key-here"

    case_file_path = "/Users/tiffanyalex/Desktop/tiffany bowhay case file"

    print("\n" + "="*80)
    print("ATTORNEY CASE FILE INGESTION")
    print("="*80)
    print(f"\nSource: {case_file_path}")
    print(f"Target: ./legal_audit_db (collection: attorney_case_file)")
    print("\nThis will create a SEPARATE searchable collection for comparison.\n")

    # Confirm
    if len(sys.argv) < 2 or sys.argv[1] != "--run":
        print("To run ingestion, execute:")
        print("  python ingest_case_file.py --run")
        print("\nOr from Python:")
        print("  from ingest_case_file import CaseFileProcessor")
        print("  processor = CaseFileProcessor()")
        print("  processor.ingest_case_file('/path/to/case/file')")
        return

    # Run ingestion
    processor = CaseFileProcessor(
        db_path="./legal_audit_db",
        collection_name="attorney_case_file"
    )

    result = processor.ingest_case_file(case_file_path)

    if result['success']:
        print("\nCase file ingested successfully!")
        print(f"You can now query this collection separately or compare with your documents.")
        if result.get('failed_files'):
            print(f"\nFailed files:")
            for f in result['failed_files']:
                print(f"  - {f}")
    else:
        print(f"\nError: {result.get('error')}")


if __name__ == "__main__":
    main()
