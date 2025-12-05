import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from retriever import retrieve_documents

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="centered"
)

# Load environment and configure API
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in .env file!")
    st.stop()

genai.configure(api_key=api_key)

# Main title
st.title("🤖 RAG Question Answering System")
st.markdown("Ask questions about Natural Language Processing and AI!")

st.divider()

# Query input
query = st.text_input("🔍 Enter your question:", placeholder="e.g., What is Natural Language Processing?")

# Search button
if st.button("Search", type="primary") and query:
    with st.spinner("🔍 Searching documents..."):
        # Retrieve documents (fixed k=3 for optimal results)
        docs = retrieve_documents(query, k=3)
        
        if not docs:
            st.error("❌ No relevant documents found in the knowledge base.")
        else:
            # Prepare context
            context = "\n\n---\n\n".join([
                f"Document {i+1}:\n{doc.page_content}" 
                for i, doc in enumerate(docs)
            ])
            
            # Create prompt
            prompt = f"""You are an expert AI assistant specializing in Natural Language Processing and AI topics.

Your task is to provide a comprehensive, detailed, and well-structured answer to the user's question based on the retrieved documents.

Retrieved Documents:
{context}

User's Question: {query}

Instructions:
1. Provide a COMPREHENSIVE and DETAILED answer using the information from the documents
2. Explain concepts thoroughly with examples when available
3. Use clear structure with:
   - An introduction to the topic
   - Main points with detailed explanations
   - Use bullet points or numbered lists for clarity
   - Include examples, definitions, and key concepts
   - A brief summary or conclusion if appropriate
4. If the documents contain related information, include it to give a complete picture
5. Write in a clear, educational style
6. Aim for a thorough, complete answer (not just a brief summary)

Please provide a detailed and comprehensive answer:"""
            
            # Generate response
            with st.spinner("🤖 Generating answer..."):
                try:
                    model = genai.GenerativeModel("models/gemini-2.5-flash")
                    
                    # Generation config for more detailed responses
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
                    
                    # Display answer
                    st.subheader("💡 Answer:")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")

# Footer
st.divider()
st.caption("Built with Streamlit, LangChain, FAISS, and Google Gemini AI")