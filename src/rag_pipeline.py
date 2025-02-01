from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch
from src.data_loader import KnowledgeRetriever
import numpy as np

MODEL_DIR = "models/rag_finetuned/"

class RAGModel:
    def __init__(self):
        """Loads the fine-tuned RAG model and FAISS-based document retrieval."""
        self.tokenizer = RagTokenizer.from_pretrained(MODEL_DIR)
        self.retriever = RagRetriever.from_pretrained(MODEL_DIR, index_name="exact")
        self.model = RagSequenceForGeneration.from_pretrained(MODEL_DIR)
        self.knowledge_retriever = KnowledgeRetriever()  # FAISS-based knowledge retrieval

    def retrieve_context(self, query):
        """Retrieve relevant documents using FAISS & Knowledge Graph."""
        relevant_docs = self.knowledge_retriever.get_relevant_docs(query)
        return " ".join(relevant_docs)

    def answer_query(self, query):
        """Generates a response using FAISS + Knowledge Graph + RAG."""
        retrieved_text = self.retrieve_context(query)
        augmented_query = query + " " + retrieved_text

        inputs = self.tokenizer(augmented_query, return_tensors="pt")
        generated = self.model.generate(**inputs)
        response = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        return {"response": response, "retrieved_docs": retrieved_text}

if __name__ == "__main__":
    rag = RAGModel()
    query = "How do I reset a robotic arm?"
    response = rag.answer_query(query)
    print(f"Generated Response: {response['response']}")
    print(f"Retrieved Documents: {response['retrieved_docs']}")
