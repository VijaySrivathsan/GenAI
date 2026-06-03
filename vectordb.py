from parser import load_all_documents
from embedding_pipeline import (create_chunks, EmbeddingModel)
from langchain_chroma import Chroma
# Vector Database Class
class VectorDatabase:
    def __init__(self):
        self.vector_store = None

    def create_vector_store(self, chunks, embedding_model):
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

    def get_vector_store(self):
        return self.vector_store

# Testing
if __name__ == "__main__":
    docs = load_all_documents()
    # Create chunks
    chunks = create_chunks(docs)
    print(f"Total chunks: {len(chunks)}")
    # Load embedding model
    embedding = EmbeddingModel()
    embedding.load_model()
    model = embedding.get_embedding_model()
    print("\nEmbedding model loaded.")
    # Create vector database
    vectordb = VectorDatabase()
    vectordb.create_vector_store(chunks, model)
    db = vectordb.get_vector_store()
    print("\nVector database created successfully.")