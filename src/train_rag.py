import json
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, Trainer, TrainingArguments
from datasets import Dataset
import os

MODEL_DIR = "models/rag_finetuned/"
DATASET_PATH = "data/train.json"

# Ensure the directory exists before saving the model
os.makedirs(MODEL_DIR, exist_ok=True)

class RAGFineTuner:
    def __init__(self, num_train_epochs=3):
        """Loads pre-trained RAG model and tokenizer for fine-tuning."""
        try:
            self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
            self.retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact")
            self.model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            print(" RAG Model Loaded Successfully")
        except Exception as e:
            print(f" Error loading RAG Model: {str(e)}")
            raise RuntimeError("Failed to load model for fine-tuning.")

        self.num_train_epochs = num_train_epochs  #  Allow dynamic epochs

    def load_dataset(self):
        """Loads the fine-tuning dataset from train.json."""
        if not os.path.exists(DATASET_PATH):
            print(f" Error: Training dataset file {DATASET_PATH} not found.")
            return None

        try:
            with open(DATASET_PATH, "r") as file:
                data = json.load(file)["data"]

            # Validate dataset format
            for sample in data:
                if not all(key in sample for key in ["question", "context", "answer"]):
                    print(f" Error: Invalid dataset format. Each entry must have 'question', 'context', and 'answer'.")
                    return None

            # Convert to Hugging Face dataset format
            dataset = Dataset.from_list([
                {
                    "question": sample["question"],
                    "context": sample["context"],
                    "answer": sample["answer"]
                }
                for sample in data
            ])
            print(f" Loaded dataset with {len(dataset)} training samples.")
            return dataset

        except json.JSONDecodeError:
            print(f" Error: Failed to decode JSON from {DATASET_PATH}.")
            return None

    def fine_tune(self):
        """Fine-tunes the RAG model using PyTorch & Hugging Face Trainer."""
        train_dataset = self.load_dataset()
        if train_dataset is None:
            print(" Error: Training aborted due to dataset issue.")
            return

        training_args = TrainingArguments(
            output_dir=MODEL_DIR,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            evaluation_strategy="epoch",
            save_total_limit=1,
            num_train_epochs=self.num_train_epochs,  #  Allow dynamic epochs
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

        print(" Starting Fine-Tuning...")
        trainer.train()
        print(" Fine-Tuning Completed.")

        # Save fine-tuned model
        self.model.save_pretrained(MODEL_DIR)
        self.tokenizer.save_pretrained(MODEL_DIR)
        print(f" Fine-tuned model saved in {MODEL_DIR}")

if __name__ == "__main__":
    fine_tuner = RAGFineTuner(num_train_epochs=5)  #  Allow user-defined training epochs
    fine_tuner.fine_tune()
