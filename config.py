import os

# Clé API OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "VOTRE_CLE_API_OPENAI")

# Modèle d'embedding
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Dossiers des données
DATA_DIR = "./data"
FAISS_INDEX_PATH = "./faiss_index"

# Paramètres de chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

