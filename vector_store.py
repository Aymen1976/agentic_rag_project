import pickle
import faiss
import numpy as np
import os
from config import FAISS_INDEX_PATH

def main():
    # Charger les embeddings
    with open("embeddings_data.pkl", "rb") as f:
        data_records = pickle.load(f)

    # Convertir en matrice NumPy
    embeddings = np.array([record["embedding"] for record in 
data_records], dtype='float32')

    # Créer l'index FAISS
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # Sauvegarder l'index
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, 
"index.faiss"))

    # Sauvegarder les chunks pour retrouver le texte
    with open(os.path.join(FAISS_INDEX_PATH, "chunks.pkl"), "wb") as f:
        pickle.dump([record["chunk"] for record in data_records], f)

    print(f"Index FAISS créé avec {len(embeddings)} vecteurs.")
    print(f"Index sauvegardé dans {FAISS_INDEX_PATH}/index.faiss")
    print(f"Chunks sauvegardés dans {FAISS_INDEX_PATH}/chunks.pkl")

if __name__ == "__main__":
    main()

