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
