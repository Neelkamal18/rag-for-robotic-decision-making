from nltk.translate.bleu_score import sentence_bleu
import numpy as np

class Evaluator:
    def __init__(self, model):
        self.model = model

    def calculate_bleu(self, reference, generated):
        """Calculates BLEU score between reference and generated responses."""
        return sentence_bleu([reference.split()], generated.split())

    def recall_at_k(self, retrieved_docs, relevant_docs, k=5):
        """Calculates Recall@K for knowledge retrieval."""
        return len(set(retrieved_docs[:k]) & set(relevant_docs)) / len(relevant_docs)

if __name__ == "__main__":
    evaluator = Evaluator(RAGModel())
    print("BLEU Score:", evaluator.calculate_bleu("Reset the robotic arm", "To reset the robotic arm, press stop."))
