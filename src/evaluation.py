import time
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from src.rag_pipeline import RAGModel
from src.data_loader import KnowledgeRetriever

class Evaluator:
    def __init__(self):
        """Initialize RAG model and FAISS-based retriever for evaluation."""
        try:
            self.model = RAGModel()
            self.retriever = KnowledgeRetriever()
            print(" Evaluator initialized successfully.")
        except Exception as e:
            print(f" Error initializing evaluator: {str(e)}")
            self.model = None
            self.retriever = None

    def calculate_bleu(self, reference, generated):
        """Computes BLEU score between reference and generated text."""
        if not generated or not reference:
            print(" Warning: Empty reference or generated text.")
            return 0.0
        return sentence_bleu([reference.split()], generated.split())

    def recall_at_k(self, query, relevant_docs, k=3):
        """Calculates Recall@K for FAISS retrieval."""
        if not self.retriever:
            print(" Error: KnowledgeRetriever is not initialized.")
            return 0.0

        retrieved_docs = self.retriever.get_relevant_docs(query, k)
        if not retrieved_docs:
            print(f"⚠️ Warning: No documents retrieved for query: '{query}'")
            return 0.0

        return len(set(retrieved_docs) & set(relevant_docs)) / len(relevant_docs)

    def measure_response_time(self, query):
        """Measures response latency of RAG inference."""
        if not self.model:
            print(" Error: RAG Model is not initialized.")
            return None

        start_time = time.time()
        self.model.answer_query(query)
        return time.time() - start_time

    def batch_evaluate(self, test_cases):
        """Runs evaluation for multiple queries."""
        for i, (query, reference_answer, relevant_docs) in enumerate(test_cases):
            print(f"\n Evaluating Query {i+1}: {query}")

            # Test BLEU Score
            generated_response = self.model.answer_query(query)["response"]
            bleu_score = self.calculate_bleu(reference_answer, generated_response)
            print(f"BLEU Score: {bleu_score:.4f}")

            # Test Recall@K
            recall = self.recall_at_k(query, relevant_docs, k=3)
            print(f"Recall@3: {recall:.4f}")

            # Test Response Latency
            latency = self.measure_response_time(query)
            print(f"Response Latency: {latency:.4f} sec")

if __name__ == "__main__":
    evaluator = Evaluator()

    test_cases = [
        ("How do I reset a robotic arm?", "Press the reset button on the control panel.", 
         ["Press the reset button on the control panel.", "Ensure the arm is in a safe position before reset."]),
        ("What is the emergency stop procedure?", "Press the emergency stop button and turn off the system.", 
         ["Press the emergency stop button.", "Ensure the system is safely powered off."]),
        ("How can I troubleshoot a sensor failure?", "Check sensor connections and recalibrate.", 
         ["Check sensor connections.", "Run a recalibration routine."])
    ]

    evaluator.batch_evaluate(test_cases)
