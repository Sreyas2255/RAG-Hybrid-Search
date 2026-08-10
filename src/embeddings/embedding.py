from langchain_huggingface import HuggingFaceEmbeddings


def create_embedding_model():
    """
    Create the Hugging Face embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings