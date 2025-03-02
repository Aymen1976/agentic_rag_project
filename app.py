import streamlit as st
from agent import RAGAgent

# Initialiser l'agent
agent = RAGAgent()

st.title("🔍 Assistant RAG : Posez vos questions !")

question = st.text_input("💡 Posez une question :")
if question:
    response = agent.answer(question)
    st.write("**Réponse :**", response)

