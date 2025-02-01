from src.rag_pipeline import RAGModel

def run_query():
    rag = RAGModel()
    while True:
        query = input("Enter a robotics-related question: ")
        if query.lower() == "exit":
            break
        response = rag.answer_query(query)
        print("Generated Response:", response["response"])
        print("Retrieved Documents:", response["retrieved_docs"])

if __name__ == "__main__":
    run_query()
