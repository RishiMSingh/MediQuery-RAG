# main.py
from rag_pipeline import run_rag_pipeline
from evaluate import run_evaluation
import subprocess

def launch_ui():
    subprocess.run(["streamlit", "run", "ui.py"])


def main():
    #run_rag_pipeline()
    #run_evaluation()
    launch_ui()

if __name__ == "__main__":
    main()