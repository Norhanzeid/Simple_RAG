import streamlit as st
from generator import generate_answer

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    layout="centered"
)

# Main title
st.title("RAG Question Answering System")
st.markdown("Ask questions about Natural Language Processing and AI!")

st.divider()

# Query input
query = st.text_input("Enter your question:")

# Search button
if st.button("Search", type="primary") and query:
    with st.spinner("Generating answer..."):
        try:
            # Generate answer using RAG
            answer = generate_answer(query, k=3)
            
            # Display answer
            st.subheader("Answer:")
            st.markdown(answer)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
# Footer
st.divider()
st.caption("Built with Streamlit, LangChain, FAISS, and Google Gemini AI")