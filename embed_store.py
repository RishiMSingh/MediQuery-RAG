# app/embed_store.py
# embed_store.py

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from text_loader import load_all_texts, chunk_text

def create_vector_store(textbook_dir, persist_dir="chroma_db"):
    print("🔹 Loading and chunking textbook content...")

    # STEP 1: LOAD
    text = load_all_texts(textbook_dir)
    print(f"📄 Loaded text length: {len(text)} characters")

    # STEP 2: CHUNK
    chunks = chunk_text(text)
    print(f"📦 Total chunks: {len(chunks)}")

    if not chunks:
        print("❌ No chunks to embed. Are you sure the files contain content?")
        return

    # STEP 3: EMBEDDINGS
    print("🧠 Creating embeddings using HuggingFace (MiniLM)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # STEP 4: VECTOR STORE
    print("📂 Creating ChromaDB vector store...")
    vectordb = Chroma.from_texts(chunks, embedding_model, persist_directory=persist_dir)

    vectordb.persist()
    print(f"✅ Vector store created and saved with {len(chunks)} chunks to '{persist_dir}'.")

if __name__ == "__main__":
    textbook_dir = "data/data_clean/textbooks/en"
    create_vector_store(textbook_dir)
