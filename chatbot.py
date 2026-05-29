from embedding_pipeline import EmbeddingModel
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
# Load Environment Variables
load_dotenv()
# Retriever Class
class Retriever:
    def __init__(self):
        self.vector_store = None
    # Load Existing ChromaDB
    def load_vector_store(self, embedding_model):
        self.vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=embedding_model
        )
    # Retrieve Relevant Chunks
    def retrieve_documents(self, query, top_k=5):
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )
        return results
    # Filter Weak Retrievals
    def filter_relevant_documents(
        self,
        retrieved_docs,
        similarity_threshold=0.45
    ):
        filtered_docs = []
        for doc, score in retrieved_docs:
            # Lower score = better match
            if score < similarity_threshold:
                filtered_docs.append((doc, score))
        return filtered_docs
    # Create Prompt
    def create_prompt(self, query, retrieved_docs):
        context = ""
        for doc, score in retrieved_docs:
            context += f"""
Similarity Score:
{score}
Content:
{doc.page_content}
Metadata:
{doc.metadata}
-----------------------------------
"""
        prompt = f"""
You are a helpful AI assistant specialized in drone research.
STRICT RULES:
1. Answer ONLY using the provided context.
2. If the context does not contain enough relevant information,
say:
"I could not find relevant information in the uploaded drone research PDFs."
3. Do NOT use outside knowledge.
4. Keep answers clear and concise.
Context:
{context}
User Question:
{query}
Answer:
"""
        return prompt
# LLM Class
class DroneChatbot:
    def __init__(self):
        self.llm = None
    # Load LLM
    def load_llm(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )
    # Generate Response
    def generate_response(self, prompt):
        response = self.llm.invoke(prompt)
        return response.content
# Main Chatbot Execution
if __name__ == "__main__":
    print("\n================================================")
    print("        DRONE RESEARCH RAG CHATBOT")
    print("================================================\n")
    print("Hello! I can answer questions based on")
    print("the uploaded drone research PDFs.\n")
    print("Type 'quit' anytime to exit.\n")
    # Load Embedding Model
    embedding = EmbeddingModel()
    embedding.load_model()
    model = embedding.get_embedding_model()
    print("Embedding model loaded successfully.\n")
    # Load Vector Database
    retrieval = Retriever()
    retrieval.load_vector_store(model)
    print("Vector database loaded successfully.\n")
    chatbot = DroneChatbot()
    chatbot.load_llm()
    print("LLM loaded successfully.\n")
    while True:
        print("\n------------------------------------------------")
        query = input("\nPlease enter your query: ")
        # Quit Condition
        if query.lower() == "quit":
            print("\nThank you for using the chatbot!")
            print("Goodbye!\n")
            break
        # Retrieve Documents
        retrieved_docs = retrieval.retrieve_documents(
            query=query,
            top_k=5
        )
        # Filter Relevant Chunks
        filtered_docs = retrieval.filter_relevant_documents(
            retrieved_docs,
            similarity_threshold=0.85
        )
        # No Relevant Information
        if len(filtered_docs) == 0:
            print("\n================================================")
            print("NO RELEVANT INFORMATION FOUND")
            print("================================================\n")
            print("I could not find relevant information")
            print("in the uploaded drone research PDFs.\n")
            continue
        # Show Retrieved Chunks
        print("\n================================================")
        print("TOP RETRIEVED CHUNKS")
        print("================================================\n")
        for i, (doc, score) in enumerate(filtered_docs):
            print(f"\nChunk {i+1}")
            print(f"\nSimilarity Distance Score: {score:.4f}")
            print("\nContent:\n")
            print(doc.page_content)
            print("\nMetadata:\n")
            print("\n" + "="*60)
        # Create Prompt
        prompt = retrieval.create_prompt(
            query,
            filtered_docs
        )
        # Generate Final Response
        response = chatbot.generate_response(prompt)
        # Final Chatbot Response
        print("\n================================================")
        print("CHATBOT RESPONSE")
        print("================================================\n")
        print(response)
        print("\n")