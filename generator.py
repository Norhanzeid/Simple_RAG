import os
from dotenv import load_dotenv
import google.generativeai as genai
from retriever import retrieve_documents, retrieve_with_scores


def setup_gemini_api():
    """Load environment variables and configure Gemini API."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found in .env file!")
    
    genai.configure(api_key=api_key)
    print("✅ Gemini API configured successfully\n")
    return api_key


def list_available_models():
    """List all available Gemini models."""
    print("📋 Available Gemini Models:")
    print("=" * 70)
    
    models = genai.list_models()
    available_models = []
    
    for m in models:
        supports_generation = "generateContent" in m.supported_generation_methods
        if supports_generation:
            available_models.append(m.name)
            print(f"✓ {m.name}")
    
    print("=" * 70 + "\n")
    return available_models


def generate_answer(query, model_name="models/gemini-2.5-flash", k=3, temperature=0.7):
    """
    Generate an answer using RAG (Retrieval-Augmented Generation).
    
    Args:
        query (str): User's question
        model_name (str): Gemini model to use
        k (int): Number of documents to retrieve
        temperature (float): Model temperature (0.0-1.0)
    
    Returns:
        dict: Contains answer, retrieved documents, and metadata
    """
    print(f"🔍 Retrieving top {k} relevant documents...")
    
    # Retrieve relevant documents
    results = retrieve_documents(query, k=k)
    
    if not results:
        return {
            "answer": "No relevant documents found in the knowledge base.",
            "documents": [],
            "query": query
        }
    
    # Prepare context from retrieved documents
    docs_text = "\n\n---\n\n".join([
        f"Document {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    print(f"✓ Retrieved {len(results)} documents")
    print(f"🤖 Generating answer using {model_name}...\n")
    
    # Create prompt for the model
    prompt = f"""You are an expert AI assistant specializing in Natural Language Processing and AI topics. 
Your task is to provide accurate, informative, and well-structured answers based on the retrieved documents.

Retrieved Documents:
{docs_text}

User's Question: {query}

Instructions:
1. Answer the question using ONLY information from the retrieved documents
2. If the documents don't contain enough information, state that clearly
3. Provide a clear, concise, and well-organized answer
4. Use bullet points or numbered lists when appropriate
5. Do not make up information that is not in the documents

Answer:"""

    # Generate response using Gemini
    model = genai.GenerativeModel(model_name)
    
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,
    }
    
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    
    return {
        "answer": response.text,
        "documents": results,
        "query": query,
        "model": model_name,
        "num_docs_retrieved": len(results)
    }


def interactive_rag_session(model_name="models/gemini-2.5-flash"):
    """Start an interactive RAG Q&A session."""
    print("\n" + "="*70)
    print("🚀 RAG Interactive Session Started")
    print("="*70)
    print("Type 'exit' or 'quit' to end the session\n")
    
    while True:
        query = input("❓ Your Question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Ending session. Goodbye!")
            break
        
        if not query:
            print("⚠️  Please enter a question.\n")
            continue
        
        print()
        result = generate_answer(query, model_name=model_name, k=3)
        
        print("="*70)
        print("💡 Answer:")
        print("="*70)
        print(result["answer"])
        print("\n" + "="*70)
        print(f"📊 Retrieved {result['num_docs_retrieved']} documents | Model: {result['model']}")
        print("="*70 + "\n")


# -----------------------------
if __name__ == "__main__":
    # Setup API
    setup_gemini_api()
    
    # List available models (optional - comment out if not needed)
    available_models = list_available_models()
    
    # Choose model from the available list above
    MODEL_NAME = "models/gemini-2.5-flash"  # Fast, efficient, latest (recommended)
    # MODEL_NAME = "models/gemini-2.5-pro"  # More capable but slower
    # MODEL_NAME = "models/gemini-flash-latest"  # Alternative fast option
    # MODEL_NAME = "models/gemini-pro-latest"  # Alternative pro option
    
    print(f"🤖 Using model: {MODEL_NAME}\n")
    print("🎯 Testing RAG System with Sample Queries\n")
    
    # Test queries
    test_queries = [
        "What is Natural Language Processing?",
        "Explain tokenization and why it's important",
        "What are transformers and how do they work?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Test Query {i}: {query}")
        print('='*70)
        
        result = generate_answer(query, model_name=MODEL_NAME, k=2)
        
        print("\n💡 Answer:")
        print("-"*70)
        print(result["answer"])
        print("-"*70)
        print(f"📊 Documents retrieved: {result['num_docs_retrieved']}")
        print('='*70)
    
    # Start interactive session (uncomment to enable)
    # print("\n")
    # interactive_rag_session(model_name=MODEL_NAME)