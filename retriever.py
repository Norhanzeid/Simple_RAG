from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
    Retrieve the top-k most similar documents with similarity scores.
    
    Args:
        query (str): The search query
        k (int): Number of documents to retrieve (default: 3)
    
    Returns:
        list: List of tuples (document, score)
    """
    vector_db = load_vector_db()
    results = vector_db.similarity_search_with_score(query, k=k)
    return results


if __name__ == "__main__":
    print("Testing Document Retrieval with Similarity Scores\n")
    
    # Test query
    query = "What is Natural Language Processing?"
    print(f"Query: {query}")
    print("=" * 70)
    
    # Retrieve documents with scores
    results = retrieve_documents(query, k=3)
    
    print(f"\nFound {len(results)} documents:\n")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"Document {i} - Similarity Score: {score:.4f}")
        print(f"Content Preview: {doc.page_content[:400]}...")
        print("-" * 70)
