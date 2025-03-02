import nltk
nltk.data.path = ["C:\\Users\\AmineBENCHOHRA.AzureAD\\AppData\\Roaming\\nltk_data"]
nltk.download('punkt')

import nltk
nltk.data.path.append("C:\\Users\\AmineBENCHOHRA.AzureAD\\AppData\\Roaming\\nltk_data")
nltk.download('punkt')

# -*- coding: utf-8 -*-
import os
import glob
import nltk
import pdfplumber  # Pour extraire du texte des PDF
import fitz  # PyMuPDF (autre méthode pour lire les PDF)
import pickle
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from config import DATA_DIR, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP

nltk.download('punkt')

def read_file(file_path):
    """Lit un fichier TXT ou PDF et retourne son contenu sous forme de texte."""
    ext = file_path.split(".")[-1].lower()
    
    if ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    elif ext == "pdf":
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        return text.strip()
    
    return ""  # Retourne une chaîne vide si le format n'est pas pris en charge

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence.split()) <= chunk_size:
            current_chunk.append(sentence)
            current_length += len(sentence.split())
        else:
            chunks.append(" ".join(current_chunk))
            overlap_tokens = current_chunk[-chunk_overlap:] if chunk_overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_tokens
            current_length = sum(len(s.split()) for s in current_chunk)
            current_chunk.append(sentence)
            current_length += len(sentence.split())

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def main():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    all_files = glob.glob(os.path.join(DATA_DIR, "*.*"))  # Accepte tous les fichiers

    data_records = []

    for file_path in all_files:
        text = read_file(file_path)  # Utilise la nouvelle fonction

        if not text.strip():
            print(f"⚠️ Impossible d'extraire du texte de {file_path}, fichier ignoré.")
            continue

        text_chunks = chunk_text(text)

        for chunk in text_chunks:
            embedding = embedding_model.encode(chunk)
            data_records.append({"chunk": chunk, "embedding": embedding})

    with open("embeddings_data.pkl", "wb") as f:
        pickle.dump(data_records, f)

    print(f"✅ Total chunks créés : {len(data_records)}")
    print("✅ Embeddings enregistrés dans embeddings_data.pkl")

if __name__ == "__main__":
    main()

