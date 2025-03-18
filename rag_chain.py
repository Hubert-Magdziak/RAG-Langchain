from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

import os

def load_documents(filepath):
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def create_vectorstore(docs, persist_dir="chroma_store"):
    embeddings = OllamaEmbeddings()
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

def get_retriever(persist_dir="chroma_store"):
    embeddings = OllamaEmbeddings(model="llama2")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vectordb.as_retriever()

def build_qa_chain():
    llm = Ollama(model="llama2")  # You can use "llama2", "phi", etc.
    retriever = get_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
