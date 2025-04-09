# app/text_loader.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_all_texts(directory):
    all_text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as file:
                all_text += file.read() + "\n\n"
    return all_text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    textbook_dir = "data/data_clean/textbooks/en"
    raw_text = load_all_texts(textbook_dir)
    chunks = chunk_text(raw_text)

    print(f"✅ Total Chunks: {len(chunks)}")
    print(f"🔹 Sample Chunk:\n{chunks[0][:500]}")
