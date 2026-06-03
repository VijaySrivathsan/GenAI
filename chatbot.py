from embedding_pipeline import EmbeddingModel
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from Voice_input import VoiceInput
from Image_Input import ImageInput
from Speech_Output import Speaker
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
        similarity_threshold=0.85
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
        STRICT RULES:
        1. Every factual statement in your answer must be directly supported by the provided context.
        2. Do NOT add definitions, explanations, assumptions, or background knowledge that are not explicitly stated in the context.
        3. If a fact is not explicitly present in the context, do not mention it.
        4. If the context contains only partial information, answer only with the available information.
        5. If the context is unrelated to the question, respond:
        "I could not find relevant information in the uploaded documents."
        6. Do not infer technical details from terminology alone.
        7. Before generating the answer, identify all facts in the context relevant to the question.
        8. Use only those identified facts when constructing the answer.
        9. Do not add definitions, explanations, or background information unless explicitly present in the context.
        10. If a statement cannot be traced directly to the context, omit it.
        11. I want only the final output to be displayed and not your thinking process
        12. The answer must be coherent
        13. Generate the answer after processing data from all the contexts.
        14. The answer must be presented in an easily readable format by the user and must cover in detail yet concisely.
        15. (For image based queries)Do not use any content from the image description for answering the query and use only the bare minimum from the image description while answering the question. Do not use any factual information given from the image description which is not present in the context extracted from the relevant chunks in the database.
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
# Load Voice Model

if __name__ == "__main__":
    print("\n================================================")
    print("        DRONE RESEARCH RAG CHATBOT")
    print("================================================\n")
    print("Hello! I can answer questions based on")
    print("the uploaded drone research materials.\n")
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
    voice = VoiceInput()
    voice.load_model()
    print("Voice model loaded successfully.\n")
    image_handler = ImageInput()
    speaker = Speaker()
    while True:
        print("\n===================================")
        print("1. Text Query")
        print("2. Voice Query")
        print("3. Image Query")
        print("4. Quit")
        print("===================================\n")
        choice = input("Choose an option: ")
        if choice == "1":
            query = input("\nEnter your query: ")
        elif choice == "2":
            voice.record_audio()
            query = voice.transcribe_audio()
            print("\nDetected Query:")
            print(query)
        elif choice == "3":
            image_path = input(
            "\nEnter image path: "
            )
            user_query = input(
            "\nEnter your query: "
            )
            image_description = image_handler.analyze_image(
            chatbot.llm,
            image_path,
            user_query
            )
            print("\nIMAGE DESCRIPTION:\n")
            print(image_description)
            query = f"""
            Image Description:
            {image_description}
            User Question:
            {user_query}
            """
        elif choice == "4":
            print("\nThank you for using the chatbot!")
            print("Goodbye!\n")
            break
        else:
            print("\nInvalid choice.")
            continue
        # Retrieve Documents
        retrieved_docs = retrieval.retrieve_documents(
            query=query,
            top_k=10
        )
        # for doc, score in retrieved_docs:
        #     print("\nSCORE:", score)
        #     print(doc.page_content[:300])
        # Filter Relevant Chunks
        filtered_docs = retrieved_docs
        # No Relevant Information
        if len(filtered_docs) == 0:
            print("\n================================================")
            print("NO RELEVANT INFORMATION FOUND")
            print("================================================\n")
            print("I could not find relevant information")
            print("in the uploaded drone research PDFs.\n")
            continue
        # Show Retrieved Chunks
        # print("\n================================================")
        # print("TOP RETRIEVED CHUNKS")
        # print("================================================\n")
        # for i, (doc, score) in enumerate(filtered_docs):
        #     print(f"\nSimilarity Distance Score: {score:.4f}")
        #     print("\nContent:\n")
        #     print(doc.page_content)
        #     print("\nMetadata:\n")
        #     print("\n" + "="*60)
        # Create Prompt
        print(f"\n")
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
        choice = input("Would you like the response to be read out aloud? (y/n): ")
        if choice.lower() == "y":
            speaker.speak(response)