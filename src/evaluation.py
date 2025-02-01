import time
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from src.rag_pipeline import RAGModel
from src.data_loader import KnowledgeRetriever

class Evaluator:
    def __init__(self):
        """Initialize RAG model and FAISS-based retriever for evaluation."""
        self.model = RAGModel()
        self.retriever = KnowledgeRetriever()

    def calculate_bleu(self, reference, generated):
        """Computes BLEU score between reference and generated text."""
        return sentence_bleu([reference.split()], generated.split())

    def recall_at_k(self, query, relevant_docs, k=3):
        """Calculates Recall@K for FAISS retrieval."""
        retrieved_docs = self.retriever.get_relevant_docs(query, k)
        return len(set(retrieved_docs) & set(relevant_docs)) / len(relevant_docs)

    def measure_response_time(self, query):
        """Measures response latency of RAG inference."""
        start_time = time.time()
        self.model.answer_query(query)
        return time.time() - start_time

if __name__ == "__main__":
    evaluator = Evaluator()
    
    test_query = "How do I reset a robotic arm?"
    reference_answer = "Press the reset button on the control panel."
    
    # Test BLEU Score
    generated_response = evaluator.model.answer_query(test_query)["response"]
    bleu_score = evaluator.calculate_bleu(reference_answer, generated_response)
    print(f"🔹 BLEU Score: {bleu_score:.4f}")

    # Test Recall@K
    relevant_docs = ["Press the reset button on the control panel.", "Ensure the arm is in a safe position before reset."]
    recall = evaluator.recall_at_k(test_query, relevant_docs, k=3)
    print(f"🔹 Recall@3: {recall:.4f}")

    # Test Response Latency
    latency = evaluator.measure_response_time(test_query)
    print(f"⚡ Response Latency: {latency:.4f} sec")
