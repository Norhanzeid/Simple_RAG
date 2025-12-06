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

Your task is to provide a COMPREHENSIVE, DETAILED, and WELL-STRUCTURED answer based on the retrieved documents.

Retrieved Documents:
{context}

User's Question: {query}

Instructions:
1. Provide a DETAILED and THOROUGH answer using ALL relevant information from the documents
2. DO NOT give brief or short answers - explain concepts fully with examples and details
3. Structure your answer with:
   - A clear introduction explaining the topic
   - Multiple detailed points with explanations
   - Use bullet points or numbered lists for clarity
   - Include examples, definitions, and key concepts from the documents
   - Provide context and background information
4. Aim for a comprehensive response (at least 150-200 words when possible)
5. Write in a clear, educational, and informative style
6. Include ALL relevant details from the retrieved documents

Please provide a detailed and comprehensive answer:"""

    # Generate response with config for longer outputs
    model = genai.GenerativeModel(model_name)
    
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
    
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    
    return response.text


if __name__ == "__main__":
    print("Testing RAG Answer Generation\n")
    
    # Test query
    query = "What is Natural Language Processing?"
    print(f"Question: {query}")
    print("=" * 70)
    
    # Generate answer
    answer = generate_answer(query, k=3)
    
    print("\n💡 Generated Answer:")
    print("=" * 70)
    print(answer)
    print("=" * 70)