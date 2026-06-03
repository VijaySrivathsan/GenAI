from langchain_community.document_loaders import PyPDFLoader
from Audio_Parser import AudioParser
import os

def load_all_documents():
    data_folder = "/Users/vijaysrivathsanbalaji/Documents/DRDO/GenAI/Data"
    all_documents = []
    # ---------------------------
    # Load all PDF files
    # ---------------------------
    pdf_files = [
        file
        for file in os.listdir(data_folder)
        if file.lower().endswith(".pdf")
    ]
    for pdf in pdf_files:
        pdf_path = os.path.join(
            data_folder,
            pdf
        )
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
    # Load all WAV files
    wav_files = [
        file
        for file in os.listdir(data_folder)
        if file.lower().endswith(".wav")
    ]
    audio_parser = AudioParser()
    audio_parser.load_model()
    for wav_file in wav_files:
        wav_path = os.path.join(
            data_folder,
            wav_file
        )
        audio_documents = audio_parser.parse_audio(
            wav_path
        )
        all_documents.extend(audio_documents)
    return all_documents

# Testing the parser(Checks whether the documents are parsed correctly) - Only for simply verifying whether whatever that has been done till now is valid
if __name__ == "__main__": #This part of code will run only when parser.py is executed and if it this file is run by importing from another file, this code block won't run as "parser" will be there in the place of "__name__"
    docs = load_all_documents()
    print(f"Total pages loaded: {len(docs)}")
    print("\nFirst document/page:\n")
    print(docs[0].page_content[:500])
    print("\nMetadata:\n")
    print(docs[0].metadata)