from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


def load_vector_db():
    """Load the FAISS vector database with embeddings."""
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_db = FAISS.load_local(
        "./data/vector_db",
        embedding,
        allow_dangerous_deserialization=True 
    )
    
    return vector_db


def retrieve_documents(query, k=3):
    """
    Retrieve the top-k most similar documents for a given query.
    
    Args:
        query (str): The search query
        k (int): Number of documents to retrieve (default: 3)
    
    Returns:
        list: List of retrieved documents
    """
    vector_db = load_vector_db()
    results = vector_db.similarity_search(query, k=k)
    return results


def retrieve_with_scores(query, k=3):
    """
    Retrieve documents with similarity scores.
    
    Args:
        query (str): The search query
        k (int): Number of documents to retrieve
    
    Returns:
        list: List of tuples (document, score)
    """
    vector_db = load_vector_db()
    results = vector_db.similarity_search_with_score(query, k=k)
    return results


# -----------------------------
if __name__ == "__main__":
    print("🔍 RAG Retriever - Testing Document Retrieval\n")
    
    # Test queries
    queries = [
        "What is Natural Language Processing?",
        "Explain tokenization in NLP",
        "What are transformers in deep learning?"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        print("=" * 70)
        
        # Retrieve top 2 documents
        results = retrieve_documents(query, k=2)
        
        for i, doc in enumerate(results, 1):
            print(f"\n📄 Document {i}:")
            print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
            print("-" * 70)
        
        print("\n")
    
    # Example with scores
    print("\n🎯 Retrieval with Similarity Scores:")
    print("=" * 70)
    query = "What is sentiment analysis?"
    results_with_scores = retrieve_with_scores(query, k=2)
    
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\n📄 Document {i} (Score: {score:.4f}):")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
        print("-" * 70)
