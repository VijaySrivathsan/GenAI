from langchain_community.document_loaders import PyPDFLoader
from Audio_Parser import AudioParser
import os

def load_all_pdfs():
    pdf_files = [
        "Paper1.pdf",
        "Paper2.pdf",
        "Paper3.pdf",
        "Paper4.pdf"
    ]
    all_documents = []
    for pdf in pdf_files:
        pdf_path = os.path.join("/Users/vijaysrivathsanbalaji/Documents/DRDO/GenAI/Data", pdf)
        loader = PyPDFLoader(pdf_path) #Preparing the parser for parsing
        documents = loader.load() #Parsing the each pdf where each page is stored as a separate Document object element in the list, containing metadata + content
        all_documents.extend(documents) #Adding the pages of the current pdf to the pages of pdfs that were already stored in the all pdf list
    audio_path = os.path.join("/Users/vijaysrivathsanbalaji/Documents/DRDO/GenAI/Data", "DroneExample.wav")
    audio_parser = AudioParser()
    audio_parser.load_model()
    audio_documents = audio_parser.parse_audio(audio_path)
    documents = all_documents + audio_documents
    return documents 

# Testing the parser(Checks whether the documents are parsed correctly) - Only for simply verifying whether whatever that has been done till now is valid
if __name__ == "__main__": #This part of code will run only when parser.py is executed and if it this file is run by importing from another file, this code block won't run as "parser" will be there in the place of "__name__"
    docs = load_all_pdfs()
    print(f"Total pages loaded: {len(docs)}")
    print("\nFirst document/page:\n")
    print(docs[0].page_content[:500])
    print("\nMetadata:\n")
    print(docs[0].metadata)