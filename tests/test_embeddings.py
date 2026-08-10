from src.embeddings.embedding import create_embedding_model


embedding_model = create_embedding_model()

text = "What is machine learning?"

vector = embedding_model.embed_query(text)

print("Embedding created successfully!")
print("Vector dimensions:", len(vector))
print("First 5 values:", vector[:5])