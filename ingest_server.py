"""
Legal Audit Server - Document Ingestion (Anthropic/HuggingFace Version)
Uses FREE HuggingFace embeddings - no API costs for document processing
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    print("\n" + "="*60)
    print("LEGAL RAG AUDIT - DOCUMENT INGESTION")
    print("Using HuggingFace Embeddings (Free, Local)")
    print("="*60 + "\n")

    # Auto-detect directory
    possible_dirs = ["./documents", "./documents"]
    directory = next((d for d in possible_dirs if os.path.exists(d)), None)

    if not directory:
        directory = "./documents"
        os.makedirs(directory)
        print(f"Created empty directory: {directory}")
        print("Please add your PDFs here and run again.")
        return

    print(f"📁 Indexing documents from: {directory}")

    # Load PDFs
    documents = []
    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            print(f"  Loading: {file}")
            try:
                loader = PyPDFLoader(os.path.join(directory, file))
                documents.extend(loader.load())
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")

    if not documents:
        print("\n✗ No PDFs found. Add your files and run again.")
        print("\nRequired files:")
        print("  - 2022 Complaint")
        print("  - 2025 Deposition (PRIORITY_TRUTH)")
        print("  - 2025 Interrogatories (PRIORITY_TRUTH)")
        print("  - Attorney correspondence")
        return

    print(f"\n✓ Successfully loaded {len(documents)} pages")

    # Split into chunks
    print("\n📝 Splitting documents into chunks for analysis...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)
    print(f"✓ Created {len(splits)} text chunks")

    # Initialize HuggingFace embeddings (FREE - runs locally)
    print("\n🤖 Loading HuggingFace embedding model...")
    print("   (First run will download model, ~90MB)")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    print("✓ Embedding model loaded")

    # Create vector database
    print("\n💾 Creating ChromaDB vector database...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./legal_audit_db"
    )

    print("\n" + "="*60)
    print("✓ RAG SERVER ONLINE")
    print(f"✓ {len(documents)} pages indexed into legal_audit_db")
    print("✓ Using HuggingFace embeddings (no API costs)")
    print("="*60)
    print("\nNext step: Run 'python3 rag_engine.py' to perform the audit")
    print("           (Requires ANTHROPIC_API_KEY)")

if __name__ == "__main__":
    main()
