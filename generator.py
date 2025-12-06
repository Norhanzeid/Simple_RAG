import os
from dotenv import load_dotenv
import google.generativeai as genai
from retriever import retrieve_documents


def generate_answer(query, model_name="models/gemini-2.5-flash", k=3):
    """Generate an answer using RAG."""
    
    # Load API key and configure
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file!")
    genai.configure(api_key=api_key)
    
    # Retrieve relevant documents
    results = retrieve_documents(query, k=k)
    
    if not results:
        return "No relevant documents found in the knowledge base."
    
    # Prepare context from retrieved documents
    context = "\n\n---\n\n".join([
        f"Document {i+1}:\n{doc.page_content}" 
        for i, (doc, score) in enumerate(results)
    ])
    
    # Create prompt
    prompt = f"""You are an expert AI assistant specializing in Natural Language Processing and AI topics.

Retrieved Documents:
{context}

User's Question: {query}

Instructions:
1. Answer using ONLY information from the retrieved documents
2. Provide a clear, concise, and well-organized answer
3. Use bullet points when appropriate
4. If information is insufficient, state that clearly

Answer:"""

    # Generate response
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    
    return response.text