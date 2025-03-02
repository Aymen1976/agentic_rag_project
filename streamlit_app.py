import streamlit as st
from agent import RAGAgent

# Initialisation de l'agent
st.title("🔍 Assistant RAG : Posez vos questions !")
st.write("Bienvenue sur votre assistant RAG basé sur FAISS.")

# Charger l'agent RAG
rag_agent = RAGAgent()

# Interface utilisateur
question = st.text_input("💡 Posez une question :")
if question:
    response = rag_agent.answer(question)
    st.write("**Réponse :**", response)
