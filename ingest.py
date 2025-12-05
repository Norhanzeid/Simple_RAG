import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document


def ingest_documents():
    """Ingest TXT and PDF files, chunk them, and save to FAISS vector database."""
    
    # Create directory for vector database
    os.makedirs("data/vector_db", exist_ok=True)
    
    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, 
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # ----------- Load TXT -----------
    print("📄 Loading TXT file...")
    try:
        txt_loader = TextLoader(
            r"C:\Users\HP\Downloads\NLP.txt",
            encoding="utf8"
        )
        txt_docs = txt_loader.load()
        txt_chunks = splitter.split_documents(txt_docs)
        documents.extend(txt_chunks)
        print(f"✓ TXT loaded: {len(txt_chunks)} chunks created")
    except Exception as e:
        print(f"⚠ Error loading TXT: {e}")

    # ----------- Load PDF using Docling -----------
    print("📑 Loading PDF file...")
    try:
        pdf_path = r"C:\Users\HP\Downloads\nlp-notes.pdf"
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        pdf_markdown = result.document.export_to_markdown()
        
        # Convert Markdown output into LangChain Document for embedding
        pdf_doc = [Document(page_content=pdf_markdown)]
        pdf_chunks = splitter.split_documents(pdf_doc)
        documents.extend(pdf_chunks)
        print(f"✓ PDF loaded: {len(pdf_chunks)} chunks created")
    except Exception as e:
        print(f"⚠ Error loading PDF: {e}")

    # ----------- Create Vector DB -----------
    if not documents:
        print("❌ No documents to process!")
        return
    
    print(f"\n🔄 Creating embeddings for {len(documents)} total chunks...")
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("💾 Building FAISS vector database...")
    vector_db = FAISS.from_documents(documents, embedding)
    vector_db.save_local("data/vector_db")

    print(f"\n✅ Ingestion Completed Successfully!")
    print(f"   - Total chunks: {len(documents)}")
    print(f"   - Saved to: data/vector_db")


if __name__ == "__main__":
    ingest_documents()

