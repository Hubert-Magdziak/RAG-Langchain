import streamlit as st
from rag_chain import load_documents, split_documents, create_vectorstore, build_qa_chain
import os

st.set_page_config(page_title="🧠 Local RAG App with Ollama")

st.title("🧠 Ask Your PDF – Locally with Ollama + LangChain")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    filepath = os.path.join("data", uploaded_file.name)
    os.makedirs("data", exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File uploaded. Processing...")

    documents = load_documents(filepath)
    docs = split_documents(documents)
    create_vectorstore(docs)

    st.success("✅ Document indexed! Ask away:")

    qa_chain = build_qa_chain()

    question = st.text_input("Enter your question:")

    if question:
        response = qa_chain.run(question)
        st.write("📎 Answer:")
        st.write(response)