from agent import RAGAgent
from config import EMBEDDING_MODEL_NAME

def main():
    # Initialiser l'agent RAG
    rag_agent = RAGAgent(embedding_model_name=EMBEDDING_MODEL_NAME)

    print("=== Agentic RAG Demo ===")
    print("Entrez votre question (ou tapez 'exit' pour quitter) :")

    while True:
        user_input = input("> ")
        if user_input.lower() in ["exit", "quit"]:
            print("Fin du programme.")
            break

        answer = rag_agent.answer(user_input)
        print(f"\nRéponse : {answer}\n")

if __name__ == "__main__":
    main()

