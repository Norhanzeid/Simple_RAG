import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

####################### INGESTION SCRIPT ########################

def ingest_documents():

    """Ingest TXT and PDF files, chunk them,Then embedding and save to FAISS vector database."""
    
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

    print("Loading TXT file...")

    try:
        txt_loader = TextLoader(
            r"C:\Users\HP\Downloads\NLP.txt",
            encoding="utf8"
        )
        txt_docs = txt_loader.load()
        txt_chunks = splitter.split_documents(txt_docs)
        documents.extend(txt_chunks)
        print(f"TXT loaded: {len(txt_chunks)} chunks created")

    except Exception as e:
        print(f"Error loading TXT: {e}")

    # ----------- Load PDF 1 using Docling -----------

    print("Loading PDF 1 (nlp-notes.pdf)...")

    try:
        pdf_path = r"C:\Users\HP\Downloads\nlp-notes.pdf"
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        pdf_markdown = result.document.export_to_markdown()
        
        # Convert Markdown output into LangChain Document for embedding
        pdf_doc = [Document(page_content=pdf_markdown, metadata={"source": "nlp-notes.pdf"})]
        pdf_chunks = splitter.split_documents(pdf_doc)
        documents.extend(pdf_chunks)
        print(f"✅ PDF 1 loaded: {len(pdf_chunks)} chunks created")

    except Exception as e:
        print(f"❌ Error loading PDF 1: {e}")

    # ----------- Load PDF 2 (Reading4-NLP.pdf) -----------

    print("Loading PDF 2 (Reading4-NLP.pdf)...")

    try:
        pdf_path2 = r"C:\Users\HP\Downloads\Reading4-NLP.pdf"
        converter2 = DocumentConverter()
        result2 = converter2.convert(pdf_path2)
        pdf_markdown2 = result2.document.export_to_markdown()
        
        # Convert Markdown output into LangChain Document for embedding
        pdf_doc2 = [Document(page_content=pdf_markdown2, metadata={"source": "Reading4-NLP.pdf"})]
        pdf_chunks2 = splitter.split_documents(pdf_doc2)
        documents.extend(pdf_chunks2)
        print(f"✅ PDF 2 loaded: {len(pdf_chunks2)} chunks created")

    except Exception as e:
        print(f"❌ Error loading PDF 2: {e}")

    # ----------- Create Vector DB -----------
    if not documents:
        print("No documents to process!")
        return
    
    ##################### Create Embeddings and Vector DB #####################

    print(f"\n Creating embeddings for {len(documents)} total chunks...")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS vector database...")
    vector_db = FAISS.from_documents(documents, embedding)
    vector_db.save_local("data/vector_db")

    print(f"\n✅ Ingestion Completed Successfully!")
    print(f"   📊 Total chunks: {len(documents)}")
    print(f"   📁 Saved to: data/vector_db")
    print(f"\n📋 Summary:")
    print(f"   - Combined all TXT and PDF files into single vector database")
    print(f"   - Ready for semantic search and question answering")


if __name__ == "__main__":
    ingest_documents()

