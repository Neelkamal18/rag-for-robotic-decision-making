from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

class RAGModel:
    def __init__(self):
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact")
        self.model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

    def answer_query(self, query):
        """Generates a response based on the retrieved knowledge."""
        inputs = self.tokenizer(query, return_tensors="pt")
        generated = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

if __name__ == "__main__":
    rag = RAGModel()
    response = rag.answer_query("How do I reset a robotic arm?")
    print(response)
