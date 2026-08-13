"""
Retrieval Evaluation

Evaluates:
    1. Dense Retrieval
    2. BM25 Retrieval
    3. Hybrid RRF Retrieval
    4. Hybrid + Cross-Encoder Reranking

Metrics:
    Recall@5
    Recall@10
    MRR

IMPORTANT:
    Ground-truth IDs are validated before evaluation.

The evaluation does NOT modify:
    - Dense retrieval
    - BM25 retrieval
    - Hybrid retrieval
    - Reranker
    - ChromaDB
    - Chunking
"""


from src.retrieval.dense_retriever import dense_search
from src.retrieval.bm25_retriever import (
    create_bm25_index,
    bm25_search,
)
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import rerank_results


# ============================================================
# Evaluation Questions
# ============================================================

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


# ============================================================
# Ground Truth
# ============================================================
#
# IMPORTANT:
# These IDs must correspond to chunks that actually contain
# the answer to the question.
#
# The evaluator will validate that every ID exists.
#
# If an ID is wrong, the evaluator will clearly report it
# instead of silently producing misleading metrics.
# ============================================================

GROUND_TRUTH = {

    "What is machine learning?": [
        "chunk_0114_000_000975",
    ],

    "What is deep learning?": [
        "chunk_0028_001_000748",
    ],

    "What is representation learning?": [
        "chunk_0023_001_000738",
    ],

    "What is a neural network?": [
        "chunk_0028_000_000747",
    ],

    "What is supervised learning?": [
        "chunk_0120_000_000997",
    ],

    "What is reinforcement learning?": [
        "chunk_0121_001_001001",
    ],

    "What are the main advantages of deep learning?": [
        "chunk_0041_001_000792",
    ],

    "How does deep learning differ from traditional machine learning?": [
        "chunk_0034_002_000771",
    ],

    "What is a machine learning algorithm?": [
        "chunk_0114_000_000975",
    ],

    "Why has deep learning become more useful over time?": [
        "chunk_0041_001_000792",
        "chunk_0026_002_000745",
    ],
}


# ============================================================
# Helper Functions
# ============================================================

def get_ids(results):
    """
    Extract chunk IDs from retrieval results.
    """

    return [
        result["id"]
        for result in results
        if "id" in result
    ]


def recall_at_k(results, relevant_ids, k):
    """
    Calculate Recall@K.

    Returns:
        1.0 -> at least one relevant chunk is in top K
        0.0 -> no relevant chunk is in top K
        None -> no ground truth
    """

    if not relevant_ids:
        return None

    retrieved_ids = set(
        get_ids(results[:k])
    )

    relevant_ids = set(
        relevant_ids
    )

    return float(
        bool(
            retrieved_ids.intersection(
                relevant_ids
            )
        )
    )


def reciprocal_rank(results, relevant_ids):
    """
    Calculate Reciprocal Rank.

    Example:

        Relevant chunk at rank 1 -> 1.0
        Relevant chunk at rank 2 -> 0.5
        Relevant chunk at rank 3 -> 0.3333
    """

    if not relevant_ids:
        return None

    relevant_ids = set(
        relevant_ids
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if result.get("id") in relevant_ids:
            return 1.0 / rank

    return 0.0


def average(values):
    """
    Safely calculate average.
    """

    if not values:
        return 0.0

    return sum(values) / len(values)


def print_metric(name, value):
    """
    Print metric safely.
    """

    if value is None:
        print(
            f"{name:<15}: N/A"
        )

    else:
        print(
            f"{name:<15}: {value:.4f}"
        )


# ============================================================
# Build Chunk Catalog
# ============================================================

def build_chunk_catalog(chunks):
    """
    Create:

        chunk_id -> chunk

    mapping.

    This allows us to validate the ground-truth IDs.
    """

    catalog = {}

    for index, chunk in enumerate(chunks):

        metadata = (
            chunk.metadata
            if chunk.metadata
            else {}
        )

        chunk_id = metadata.get(
            "chunk_id"
        )

        if chunk_id is None:
            chunk_id = f"chunk_{index}"

        catalog[chunk_id] = chunk

    return catalog


# ============================================================
# Validate Ground Truth
# ============================================================

def validate_ground_truth(
    ground_truth,
    chunk_catalog,
):
    """
    Validate every ground-truth chunk ID.

    This prevents invalid chunk IDs from silently
    producing misleading evaluation results.
    """

    print()
    print("=" * 70)
    print("GROUND-TRUTH VALIDATION")
    print("=" * 70)

    valid_count = 0
    invalid_count = 0

    for question in EVALUATION_QUESTIONS:

        ids = ground_truth.get(
            question,
            [],
        )

        print()
        print(
            f"Question: {question}"
        )

        if not ids:

            print(
                "  WARNING: No ground-truth IDs."
            )

            invalid_count += 1

            continue

        for chunk_id in ids:

            if chunk_id in chunk_catalog:

                print(
                    f"  [VALID]   {chunk_id}"
                )

                valid_count += 1

            else:

                print(
                    f"  [INVALID] {chunk_id}"
                )

                invalid_count += 1

    print()
    print("-" * 70)

    print(
        f"Valid ground-truth IDs   : {valid_count}"
    )

    print(
        f"Invalid ground-truth IDs : {invalid_count}"
    )

    print("-" * 70)

    return invalid_count == 0


# ============================================================
# Print Ground Truth Chunk
# ============================================================

def print_ground_truth_chunk(
    question,
    relevant_ids,
    chunk_catalog,
):
    """
    Print the actual ground-truth text.

    This is extremely useful for checking whether
    the selected chunk really answers the question.
    """

    print()
    print(
        "GROUND-TRUTH CONTENT"
    )

    for chunk_id in relevant_ids:

        chunk = chunk_catalog.get(
            chunk_id
        )

        if chunk is None:
            continue

        print()
        print(
            f"Chunk ID: {chunk_id}"
        )

        print(
            "-" * 60
        )

        text = chunk.page_content

        print(
            text[:1000]
        )

        print(
            "-" * 60
        )


# ============================================================
# Main Evaluation
# ============================================================

def run_evaluation():

    print()
    print("=" * 70)
    print("                 RETRIEVAL EVALUATION")
    print("=" * 70)

    # ========================================================
    # 1. Create BM25 index
    # ========================================================

    print()
    print("Creating BM25 index...")

    bm25, chunks = (
        create_bm25_index()
    )

    print()
    print(
        f"BM25 index ready."
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    # ========================================================
    # 2. Build chunk catalog
    # ========================================================

    chunk_catalog = build_chunk_catalog(
        chunks
    )

    print(
        f"Chunk catalog entries: "
        f"{len(chunk_catalog)}"
    )

    # ========================================================
    # 3. Validate ground truth
    # ========================================================

    ground_truth_valid = (
        validate_ground_truth(
            GROUND_TRUTH,
            chunk_catalog,
        )
    )

    # ========================================================
    # IMPORTANT
    # ========================================================
    #
    # If ground-truth IDs are invalid, STOP.
    #
    # We do NOT calculate fake evaluation numbers.
    # ========================================================

    if not ground_truth_valid:

        print()
        print("=" * 70)
        print("GROUND-TRUTH ERROR")
        print("=" * 70)

        print()
        print(
            "Some ground-truth chunk IDs do not exist "
            "in the current chunk collection."
        )

        print()
        print(
            "Evaluation stopped to prevent misleading metrics."
        )

        print()
        print(
            "The retrieval pipeline itself was NOT changed."
        )

        return

    # ========================================================
    # 4. Metric storage
    # ========================================================

    dense_recall_5 = []
    dense_recall_10 = []
    dense_mrr = []

    bm25_recall_5 = []
    bm25_recall_10 = []
    bm25_mrr = []

    hybrid_recall_5 = []
    hybrid_recall_10 = []
    hybrid_mrr = []

    reranker_recall_5 = []
    reranker_mrr = []

    # ========================================================
    # 5. Evaluate questions
    # ========================================================

    for question_number, question in enumerate(
        EVALUATION_QUESTIONS,
        start=1,
    ):

        print()
        print("#" * 70)

        print(
            f"QUESTION "
            f"{question_number}/"
            f"{len(EVALUATION_QUESTIONS)}"
        )

        print("#" * 70)

        print()
        print("Question:")
        print(question)

        relevant_ids = GROUND_TRUTH.get(
            question,
            [],
        )

        print()
        print(
            "Ground-truth chunk IDs:"
        )

        for chunk_id in relevant_ids:

            print(
                f"  - {chunk_id}"
            )

        # ----------------------------------------------------
        # Show actual ground-truth content
        # ----------------------------------------------------

        print_ground_truth_chunk(
            question,
            relevant_ids,
            chunk_catalog,
        )

        # ====================================================
        # 1. Dense Retrieval
        # ====================================================

        print()
        print("-" * 70)
        print("1. DENSE RETRIEVAL")
        print("-" * 70)

        try:

            dense_results = dense_search(
                question,
                top_k=10,
            )

        except Exception as error:

            print(
                f"Dense retrieval error: {error}"
            )

            dense_results = []

        print(
            f"Dense results: "
            f"{len(dense_results)}"
        )

        # ====================================================
        # 2. BM25 Retrieval
        # ====================================================

        print()
        print("-" * 70)
        print("2. BM25 RETRIEVAL")
        print("-" * 70)

        try:

            # IMPORTANT:
            # bm25_search signature is:
            #
            # bm25_search(
            #     bm25,
            #     chunks,
            #     query,
            #     top_k
            # )

            bm25_results = bm25_search(
                bm25,
                chunks,
                question,
                top_k=10,
            )

        except Exception as error:

            print(
                f"BM25 retrieval error: {error}"
            )

            bm25_results = []

        print(
            f"BM25 results: "
            f"{len(bm25_results)}"
        )

        # ====================================================
        # 3. Hybrid Retrieval
        # ====================================================

        print()
        print("-" * 70)
        print("3. HYBRID RRF RETRIEVAL")
        print("-" * 70)

        try:

            hybrid_results = hybrid_search(
                question,
                top_k=10,
            )

        except Exception as error:

            print(
                f"Hybrid retrieval error: {error}"
            )

            hybrid_results = []

        print(
            f"Hybrid results: "
            f"{len(hybrid_results)}"
        )

        # ====================================================
        # 4. Cross-Encoder Reranking
        # ====================================================

        print()
        print("-" * 70)
        print("4. CROSS-ENCODER RERANKING")
        print("-" * 70)

        try:

            reranked_results = rerank_results(
                question,
                hybrid_results,
                top_k=5,
            )

        except Exception as error:

            print(
                f"Reranker error: {error}"
            )

            reranked_results = []

        print(
            f"Reranked results: "
            f"{len(reranked_results)}"
        )

        # ====================================================
        # Calculate Dense Metrics
        # ====================================================

        dense_r5 = recall_at_k(
            dense_results,
            relevant_ids,
            5,
        )

        dense_r10 = recall_at_k(
            dense_results,
            relevant_ids,
            10,
        )

        dense_rr = reciprocal_rank(
            dense_results,
            relevant_ids,
        )

        if dense_r5 is not None:
            dense_recall_5.append(
                dense_r5
            )

        if dense_r10 is not None:
            dense_recall_10.append(
                dense_r10
            )

        if dense_rr is not None:
            dense_mrr.append(
                dense_rr
            )

        # ====================================================
        # Calculate BM25 Metrics
        # ====================================================

        bm25_r5 = recall_at_k(
            bm25_results,
            relevant_ids,
            5,
        )

        bm25_r10 = recall_at_k(
            bm25_results,
            relevant_ids,
            10,
        )

        bm25_rr = reciprocal_rank(
            bm25_results,
            relevant_ids,
        )

        if bm25_r5 is not None:
            bm25_recall_5.append(
                bm25_r5
            )

        if bm25_r10 is not None:
            bm25_recall_10.append(
                bm25_r10
            )

        if bm25_rr is not None:
            bm25_mrr.append(
                bm25_rr
            )

        # ====================================================
        # Calculate Hybrid Metrics
        # ====================================================

        hybrid_r5 = recall_at_k(
            hybrid_results,
            relevant_ids,
            5,
        )

        hybrid_r10 = recall_at_k(
            hybrid_results,
            relevant_ids,
            10,
        )

        hybrid_rr = reciprocal_rank(
            hybrid_results,
            relevant_ids,
        )

        if hybrid_r5 is not None:
            hybrid_recall_5.append(
                hybrid_r5
            )

        if hybrid_r10 is not None:
            hybrid_recall_10.append(
                hybrid_r10
            )

        if hybrid_rr is not None:
            hybrid_mrr.append(
                hybrid_rr
            )

        # ====================================================
        # Calculate Reranker Metrics
        # ====================================================

        reranker_r5 = recall_at_k(
            reranked_results,
            relevant_ids,
            5,
        )

        reranker_rr = reciprocal_rank(
            reranked_results,
            relevant_ids,
        )

        if reranker_r5 is not None:
            reranker_recall_5.append(
                reranker_r5
            )

        if reranker_rr is not None:
            reranker_mrr.append(
                reranker_rr
            )

        # ====================================================
        # Print Question Metrics
        # ====================================================

        print()
        print("=" * 70)
        print("QUESTION METRICS")
        print("=" * 70)

        print()
        print("Dense Retrieval")

        print_metric(
            "Recall@5",
            dense_r5,
        )

        print_metric(
            "Recall@10",
            dense_r10,
        )

        print_metric(
            "MRR",
            dense_rr,
        )

        print()
        print("BM25 Retrieval")

        print_metric(
            "Recall@5",
            bm25_r5,
        )

        print_metric(
            "Recall@10",
            bm25_r10,
        )

        print_metric(
            "MRR",
            bm25_rr,
        )

        print()
        print("Hybrid Retrieval")

        print_metric(
            "Recall@5",
            hybrid_r5,
        )

        print_metric(
            "Recall@10",
            hybrid_r10,
        )

        print_metric(
            "MRR",
            hybrid_rr,
        )

        print()
        print("Cross-Encoder Reranker")

        print_metric(
            "Recall@5",
            reranker_r5,
        )

        print_metric(
            "MRR",
            reranker_rr,
        )

    # ========================================================
    # Final Results
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("                 FINAL RETRIEVAL RESULTS")
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Calculate averages
    # --------------------------------------------------------

    dense_r5_avg = average(
        dense_recall_5
    )

    dense_r10_avg = average(
        dense_recall_10
    )

    dense_mrr_avg = average(
        dense_mrr
    )

    bm25_r5_avg = average(
        bm25_recall_5
    )

    bm25_r10_avg = average(
        bm25_recall_10
    )

    bm25_mrr_avg = average(
        bm25_mrr
    )

    hybrid_r5_avg = average(
        hybrid_recall_5
    )

    hybrid_r10_avg = average(
        hybrid_recall_10
    )

    hybrid_mrr_avg = average(
        hybrid_mrr
    )

    reranker_r5_avg = average(
        reranker_recall_5
    )

    reranker_mrr_avg = average(
        reranker_mrr
    )

    # ========================================================
    # Results Table
    # ========================================================

    print(
        f"{'Method':<25}"
        f"{'Recall@5':>12}"
        f"{'Recall@10':>12}"
        f"{'MRR':>12}"
    )

    print("-" * 70)

    print(
        f"{'Dense':<25}"
        f"{dense_r5_avg:>12.4f}"
        f"{dense_r10_avg:>12.4f}"
        f"{dense_mrr_avg:>12.4f}"
    )

    print(
        f"{'BM25':<25}"
        f"{bm25_r5_avg:>12.4f}"
        f"{bm25_r10_avg:>12.4f}"
        f"{bm25_mrr_avg:>12.4f}"
    )

    print(
        f"{'Hybrid RRF':<25}"
        f"{hybrid_r5_avg:>12.4f}"
        f"{hybrid_r10_avg:>12.4f}"
        f"{hybrid_mrr_avg:>12.4f}"
    )

    print(
        f"{'Cross-Encoder':<25}"
        f"{reranker_r5_avg:>12.4f}"
        f"{'N/A':>12}"
        f"{reranker_mrr_avg:>12.4f}"
    )

    print()
    print("=" * 70)

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("Evaluation Summary")
    print("-" * 70)

    print(
        f"Questions evaluated : "
        f"{len(EVALUATION_QUESTIONS)}"
    )

    print(
        f"Ground-truth sets   : "
        f"{len(GROUND_TRUTH)}"
    )

    print(
        f"Indexed chunks      : "
        f"{len(chunks)}"
    )

    print()
    print(
        "Evaluation completed successfully."
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_evaluation()