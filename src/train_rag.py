import json
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, Trainer, TrainingArguments
from datasets import Dataset
import os

MODEL_DIR = "models/rag_finetuned/"

# Ensure the directory exists before saving the model
os.makedirs(MODEL_DIR, exist_ok=True)

DATASET_PATH = "data/train.json"

class RAGFineTuner:
    def __init__(self):
        """Loads pre-trained RAG model and tokenizer for fine-tuning."""
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact")
        self.model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_dataset(self):
        """Loads the fine-tuning dataset from train.json."""
        with open(DATASET_PATH, "r") as file:
            data = json.load(file)["data"]
        
        # Convert to Hugging Face dataset format
        dataset = Dataset.from_list([
            {
                "question": sample["question"],
                "context": sample["context"],
                "answer": sample["answer"]
            }
            for sample in data
        ])
        return dataset

    def fine_tune(self):
        """Fine-tunes the RAG model using PyTorch & Hugging Face Trainer."""
        train_dataset = self.load_dataset()
        
        training_args = TrainingArguments(
            output_dir=MODEL_DIR,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            evaluation_strategy="epoch",
            save_total_limit=1,
            num_train_epochs=3,
            save_steps=500,
            logging_steps=100,
            logging_dir="logs",
            report_to="none",
            fp16=torch.cuda.is_available()
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )

        trainer.train()

        # Save fine-tuned model
        self.model.save_pretrained(MODEL_DIR)
        self.tokenizer.save_pretrained(MODEL_DIR)
        print(f"Fine-tuned model saved in {MODEL_DIR}")

if __name__ == "__main__":
    fine_tuner = RAGFineTuner()
    fine_tuner.fine_tune()
