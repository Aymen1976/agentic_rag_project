# -*- coding: utf-8 -*-
import faiss
import pickle
import numpy as np
import os
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME
from sentence_transformers import SentenceTransformer

class RAGAgent:
    def __init__(self, embedding_model_name=EMBEDDING_MODEL_NAME):
        print("🔹 Initialisation de l'agent RAG...")

        # Charger le modèle d'embedding
        self.model = SentenceTransformer(embedding_model_name)
        print(f"✅ Modèle Sentence-Transformers chargé : {embedding_model_name}")

        # Charger l'index FAISS et les chunks
        self.index, self.chunks = self.load_faiss_index()

    def load_faiss_index(self):
        """Charge l'index FAISS et les chunks de texte sauvegardés"""
        index_path = os.path.join(FAISS_INDEX_PATH, "index.faiss")
        chunks_path = os.path.join(FAISS_INDEX_PATH, "chunks.pkl")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise ValueError("❌ L'index FAISS est introuvable ! Veuillez exécuter `python3 vector_store.py`.")

        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        # Vérification de la taille de l'index
        if index.ntotal == 0:
            raise ValueError("❌ L'index FAISS est vide ! Exécutez `python3 vector_store.py` pour le recréer.")

        return index, chunks

    def search(self, query, top_k=3):
        """Effectue une recherche FAISS et filtre les doublons"""
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)  # Correction

        distances, indices = self.index.search(query_embedding, top_k)

        # Récupérer les résultats et supprimer les doublons
        unique_results = list(dict.fromkeys(
            [self.chunks[i] for i in indices[0] if 0 <= i < len(self.chunks)]
        ))

        return unique_results

    def answer(self, query):
        """Retourne la meilleure réponse pour une requête donnée"""
        results = self.search(query, top_k=1)
        return results[0] if results else "❌ Aucune réponse trouvée."
