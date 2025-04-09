from dotenv import load_dotenv
load_dotenv()

# rag_pipeline.py

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI  


# Load .env
load_dotenv()

def run_rag_pipeline(query):
    print("🔄 Loading vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="/Users/rishisingh/Documents/FinQuery/chroma_db", embedding_function=embedding_model)

    print("💬 Initializing OpenAI LLM...")
    llm = ChatOpenAI(
    temperature=0.2,
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

    print("🔍 Building RAG chain...")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    print("📥 Querying...")
    cot_prompt = (
    "You are a medical expert answering a USMLE-style question.\n"
    "Think step-by-step, explain your reasoning using the provided context, "
    "and then give your final answer.\n\n"
    f"Question: {query}" )
    response = qa({"query": cot_prompt})

    print("\n🧠 Answer:")
    print(response["result"])

    print("\n📚 Sources:")
    for doc in response["source_documents"]:
        print("—", doc.page_content[:100].replace("\n", " "), "...\n") 

if __name__ == "__main__":
    question = "A patient presents with fatigue, low blood pressure, and hyperpigmentation. What is the likely diagnosis?"
    run_rag_pipeline(question)
