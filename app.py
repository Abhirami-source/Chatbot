import streamlit as st
import google.generativeai as genai

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
import os

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.5-flash")

# Load and preprocess CSV
@st.cache_resource
def load_documents():
    df = pd.read_csv("stellchatbotver1(in).csv")
    df['question'] = df['question'].fillna("")
    df['answer'] = df['answer'].fillna("")
    df['type'] = df['type'].fillna("general")
    df['combined_text'] = "Question: " + df['question'] + " Answer: " + df['answer']
    documents = [str(doc) for doc in df['combined_text'].tolist() if doc.strip() != ""]
    metadatas = [{"category": str(row["type"])} for _, row in df.iterrows()]
    ids = [str(i) for i in range(len(documents))]
    return documents, metadatas, ids

# Setup ChromaDB and load collection
@st.cache_resource
def load_collection():
    documents, metadatas, ids = load_documents()
    gemini_ef = GoogleGenerativeAiEmbeddingFunction(
    api_key=st.secrets["GEMINI_API_KEY"],
    model_name="models/gemini-embedding-001"
)

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="club_collection_v0",
        embedding_function=gemini_ef
    )
    if len(collection.get()["ids"]) == 0:
        collection.add(documents=documents, ids=ids, metadatas=metadatas)
    return collection

collection = load_collection()

# Query function
def ask_stellaria(question):
    results = collection.query(query_texts=[question], n_results=2)
    if not results["documents"][0]:
        return "Sorry, I couldn't find anything relevant. Try rephrasing or ask a club member."

    context = "\n".join(results["documents"][0])
    prompt = f"""
    You are the Stellaria Club Assistant. Stellaria and club are synonymous.
    Use the context provided to answer the question.
    If the answer isn't in the context, say you don't know and suggest contacting a member.

    Context from CSV:
    {context}

    User Question: {question}
    """
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.set_page_config(page_title="Stellaria Club Assistant", page_icon="🌌")
st.title("🌌 Stellaria Club Assistant")

user_question = st.text_input("Ask Stellaria a question:")
if st.button("Submit"):
    if user_question.strip():
        with st.spinner("Thinking..."):
            answer = ask_stellaria(user_question)
            st.write("**Answer:**", answer)


