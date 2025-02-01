from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch
from src.data_loader import KnowledgeRetriever
import numpy as np
import os

MODEL_DIR = "models/rag_finetuned/"

class RAGModel:
    def __init__(self):
        """Loads the fine-tuned RAG model and FAISS-based document retrieval."""
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Error: Model directory {MODEL_DIR} not found. Ensure fine-tuning is completed.")

        try:
            self.tokenizer = RagTokenizer.from_pretrained(MODEL_DIR)
            self.retriever = RagRetriever.from_pretrained(MODEL_DIR, index_name="exact")
            self.model = RagSequenceForGeneration.from_pretrained(MODEL_DIR)
            self.knowledge_retriever = KnowledgeRetriever()  # FAISS-based knowledge retrieval
            print("RAG Model Loaded Successfully")
        except Exception as e:
            raise RuntimeError(f"Error loading RAG model: {str(e)}")

    def retrieve_context(self, query):
        """Retrieve relevant documents using FAISS & Knowledge Graph."""
        try:
            relevant_docs = self.knowledge_retriever.get_relevant_docs(query)
            if not relevant_docs:
                print("⚠️ No relevant documents found. Proceeding with query only.")
                return ""
            return " ".join(relevant_docs)
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return ""

    def answer_query(self, query):
        """Generates a response using FAISS + Knowledge Graph + RAG."""
        retrieved_text = self.retrieve_context(query)
        augmented_query = query + " " + retrieved_text if retrieved_text else query

        inputs = self.tokenizer(augmented_query, return_tensors="pt")
        generated = self.model.generate(**inputs)
        response = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        return {"response": response, "retrieved_docs": retrieved_text}

    def batch_answer_queries(self, queries):
        """Processes multiple queries in batch."""
        responses = []
        for query in queries:
            responses.append(self.answer_query(query))
        return responses

if __name__ == "__main__":
    rag = RAGModel()
    query = "How do I reset a robotic arm?"
    response = rag.answer_query(query)

    print(f"Generated Response: {response['response']}")
    print(f"Retrieved Documents: {response['retrieved_docs']}")
