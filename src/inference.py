from rag_pipeline import RAGModel

def run_query():
    rag = RAGModel()
    while True:
        query = input("Enter your robotics-related question: ")
        if query.lower() == "exit":
            break
        response = rag.answer_query(query)
        print("Response:", response)

if __name__ == "__main__":
    run_query()
