"""
RAG Evaluation Script

Evaluates the complete RAG pipeline:

PDFs
  ↓
BM25
  ↓
Dense Retrieval
  ↓
Hybrid RRF
  ↓
Reranker
  ↓
Groq
  ↓
Final Answer
"""

from src.retrieval.bm25_retriever import create_bm25_index
from src.generation.rag_pipeline import rag_answer


# --------------------------------------------------
# Evaluation questions
# --------------------------------------------------

EVALUATION_QUESTIONS = [
    "What is machine learning?",
    "What is deep learning?",
    "What is representation learning?",
    "What is a neural network?",
    "What is supervised learning?",
    "What is reinforcement learning?",
    "What are the main advantages of deep learning?",
    "How does deep learning differ from traditional machine learning?",
    "What is a machine learning algorithm?",
    "Why has deep learning become more useful over time?",
]


# --------------------------------------------------
# Run evaluation
# --------------------------------------------------

def run_evaluation():
    print("\n========================================")
    print("        RAG EVALUATION")
    print("========================================")

    print("\nCreating BM25 index...")

    bm25, chunks = create_bm25_index()

    print("\nBM25 index ready.")

    total_questions = len(EVALUATION_QUESTIONS)

    correct = 0
    incorrect = 0
    unsure = 0

    results = []

    # --------------------------------------------------
    # Evaluate each question
    # --------------------------------------------------

    for index, question in enumerate(
        EVALUATION_QUESTIONS,
        start=1,
    ):

        print("\n")
        print("========================================")
        print(f"QUESTION {index}/{total_questions}")
        print("========================================")

        print(f"\nQuestion: {question}")

        try:

            answer = rag_answer(
                question,
                bm25,
                chunks,
            )

        except Exception as error:

            print("\nERROR:")
            print(error)

            answer = (
                "ERROR: RAG pipeline failed."
            )

        print("\n----------------------------------------")
        print("GENERATED ANSWER")
        print("----------------------------------------")

        print(answer)

        # --------------------------------------------------
        # Manual evaluation
        # --------------------------------------------------

        while True:

            evaluation = input(
                "\nIs this answer "
                "(c)orrect, (i)ncorrect, or (u)nsure? "
            ).strip().lower()

            if evaluation in {
                "c",
                "i",
                "u",
            }:
                break

            print(
                "Please enter c, i, or u."
            )

        if evaluation == "c":
            correct += 1

        elif evaluation == "i":
            incorrect += 1

        else:
            unsure += 1

        results.append(
            {
                "question": question,
                "answer": answer,
                "evaluation": evaluation,
            }
        )

    # --------------------------------------------------
    # Calculate score
    # --------------------------------------------------

    evaluated_questions = (
        correct + incorrect
    )

    if evaluated_questions > 0:

        accuracy = (
            correct
            / evaluated_questions
        ) * 100

    else:

        accuracy = 0.0

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print("\n\n")
    print("========================================")
    print("          EVALUATION RESULTS")
    print("========================================")

    print(
        f"\nTotal questions : {total_questions}"
    )

    print(
        f"Correct         : {correct}"
    )

    print(
        f"Incorrect       : {incorrect}"
    )

    print(
        f"Unsure          : {unsure}"
    )

    print(
        f"\nAccuracy        : {accuracy:.2f}%"
    )

    print("\n========================================")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    run_evaluation()