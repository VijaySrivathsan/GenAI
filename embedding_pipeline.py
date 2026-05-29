from parser import load_all_pdfs
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Chunking Function

def create_chunks(documents):
    splitter = TokenTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    ) #Each chunk will contain 200 tokens with a possible overlap of not more than 50 tokens 
    chunked_documents = splitter.split_documents(documents)
    return chunked_documents

# Embedding Class

class EmbeddingModel:
    def __init__(self):
        self.embedding_model = None
    def load_model(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    def get_embedding_model(self):
        return self.embedding_model

# Embedding Creation Function

def create_embeddings(chunks, embedding_model):
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(chunk.page_content)
    embeddings = embedding_model.embed_documents(chunk_texts)
    return embeddings

# Testing

if __name__ == "__main__":
    # Load parsed documents
    docs = load_all_pdfs()
    # Create chunks
    chunks = create_chunks(docs)
    print(f"Total chunks created: {len(chunks)}")
    print("\nFirst Chunk:\n")
    print(chunks[0].page_content)
    print("\nMetadata:\n")
    print(chunks[0].metadata)

    # Load embedding model
    embedding = EmbeddingModel()
    embedding.load_model()
    model = embedding.get_embedding_model()
    print("\nEmbedding model loaded successfully.")

    # Create embeddings
    embeddings = create_embeddings(chunks, model)
    print(f"\nTotal embeddings created: {len(embeddings)}")
    print("\nFirst Embedding Vector:\n")
    print(embeddings[0][:10])