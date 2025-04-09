import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

st.set_page_config(page_title="🩺 MedRAG Assistant", layout="centered")

# Inject CSS for styling
st.markdown("""
<style>
body {
    background-color: #111111;
    color: #eeeeee;
}
.css-18e3th9 {
    padding: 2rem;
}
h1 {
    font-size: 2.5rem;
    font-weight: 800;
    color: #2ec4b6;
}
.block-container {
    padding: 2rem;
}
.stTextArea textarea {
    font-family: monospace;
    font-size: 1rem;
}
.result-box {
    background-color: #1e1e1e;
    border: 1px solid #333333;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline(model_choice):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

    llm = ChatOpenAI(
        temperature=0.2,
        model_name=model_choice,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    return qa

st.title("🩺 MedRAG: USMLE Medical Assistant")
st.markdown("Ask a USMLE-style medical question and get a contextual, step-by-step answer grounded in real textbooks.")

model_choice = st.selectbox("🔧 Choose LLM:", ["gpt-3.5-turbo", "gpt-4"])
question = st.text_area("📝 Enter your medical question:", height=180)

if st.button("💡 Get Answer") and question:
    with st.spinner("🧠 Thinking like a USMLE doctor..."):
        rag_chain = load_pipeline(model_choice)

        prompt = (
            "You are a USMLE-trained medical expert.\n"
            "Think step-by-step using the retrieved context.\n"
            "Then select the best answer from A to E.\n\n"
            f"Question: {question}\n\n"
            "Think carefully and conclude with: Final Answer: <letter>"
        )

        result = rag_chain({"query": prompt})
        answer = result["result"]
        docs = result["source_documents"]

        st.markdown("### 🧠 Answer")
        st.markdown(f"<div class='result-box'>{answer}</div>", unsafe_allow_html=True)

        st.markdown("### 📚 Retrieved Context")
        for i, doc in enumerate(docs):
            st.markdown(f"<div class='result-box'><strong>Chunk {i+1}</strong><br>{doc.page_content[:500]}...</div>", unsafe_allow_html=True)
