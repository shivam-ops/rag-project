from src.rag_system import RAGSystem


def main():
    # Initialize RAG system
    rag = RAGSystem()

    try:
        # Load and process the document
        rag.load_and_process_document()

        print("\nRAG System Ready! Type 'quit' to exit or 'clear' to clear history")

        while True:
            question = input("\nEnter your question: ")

            if question.lower() == 'quit':
                break

            if question.lower() == 'clear':
                rag.clear_chat_history()
                print("Chat history cleared!")
                continue

            try:
                response = rag.query(question)
                print("\nAnswer:", response["answer"])
                print("\nContext used:", response["context"])
            except Exception as e:
                print(f"Error processing question: {str(e)}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()