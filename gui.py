# -*- coding: utf-8 -*-
import tkinter as tk
from agent import RAGAgent
from config import EMBEDDING_MODEL_NAME

# Initialisation de l'agent RAG
rag_agent = RAGAgent(embedding_model_name=EMBEDDING_MODEL_NAME)

def search():
    """Effectue une recherche et affiche la réponse."""
    query = entry.get()
    answer = rag_agent.answer(query)
    output_label.config(text=f"Réponse :\n{answer}")

# Création de la fenêtre principale
root = tk.Tk()
root.title("Agent RAG - Interface Locale")
root.geometry("600x400")

# Zone de texte pour la question
entry_label = tk.Label(root, text="Entrez votre question :", 
font=("Arial", 12))
entry_label.pack(pady=5)

entry = tk.Entry(root, width=50, font=("Arial", 12))
entry.pack(pady=5)

# Bouton de recherche
search_button = tk.Button(root, text="Rechercher", command=search, 
font=("Arial", 12), bg="blue", fg="white")
search_button.pack(pady=10)

# Zone d'affichage de la réponse
output_label = tk.Label(root, text="Réponse : ", wraplength=500, 
justify="left", font=("Arial", 12), fg="black")
output_label.pack(pady=10)

# Lancer l'interface
root.mainloop()

