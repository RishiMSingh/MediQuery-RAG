
import os
import json
import jsonlines
import random
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

def load_medqa(file_path, num_samples=100):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return random.sample(data, num_samples)

def setup_rag():
    vectordb = Chroma(
        persist_directory="/Users/rishisingh/Documents/FinQuery/chroma_db",
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    llm = ChatOpenAI(
        temperature=0.2,
        model_name="gpt-4",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=False
    )
    return qa

def extract_final_answer(text):
    """Extract final answer in form 'Final Answer: <X>'"""
    match = re.search(r'Final Answer:\s*([A-E])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def run_evaluation():
    dataset_path = "data_clean/questions/US/test.jsonl"
    dataset = load_medqa(dataset_path, num_samples=100)
    qa_chain = setup_rag()

    results = []
    correct = 0

    for i, item in enumerate(dataset):
        question = item["question"]
        options = item["options"]
        ground_truth = item["answer_idx"]

        # Build CoT prompt
        prompt = (
            "You are a USMLE-trained medical expert.\n"
            "Think step-by-step using the context below.\n"
            "Then select the best answer from A to E.\n\n"
            f"Question: {question}\n\n"
            "Options:\n" +
            "\n".join([f"{k}. {v}" for k, v in options.items()]) +
            "\n\nThink carefully and conclude with: Final Answer: <letter>"
        )

        try:
            result = qa_chain({"query": prompt})
            full_answer = result["result"]
            model_choice = extract_final_answer(full_answer)

            is_correct = (model_choice == ground_truth)
            if is_correct:
                correct += 1

            results.append({
                "question": question,
                "options": options,
                "ground_truth": ground_truth,
                "model_answer": full_answer.strip(),
                "final_choice": model_choice,
                "is_correct": is_correct
            })

            print(f"\n🧠 Q{i+1}: {question}")
            print(f"✅ Correct: {ground_truth} | 🤖 Predicted: {model_choice} | ✅ Match: {is_correct}")
            print(f"🧾 Answer Excerpt: {full_answer[:300]}...")
            print("-" * 80)

        except Exception as e:
            print(f"❌ Error on Q{i+1}: {e}")

    print(f"\n✅ Evaluation complete — Accuracy: {correct}/{len(dataset)} = {correct / len(dataset) * 100:.1f}%")

    with open("medrag_eval_output.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_evaluation()
