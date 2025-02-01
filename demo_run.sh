#!/bin/bash

LOG_DIR="logs"
LOG_FILE="$LOG_DIR/demo_run.log"

# Create logs directory if it doesn't exist
mkdir -p $LOG_DIR

echo "🚀 Starting RAG for Robotic Decision-Making Demo 🚀" | tee $LOG_FILE

# 1️⃣ Install Dependencies
echo "📦 Checking and installing dependencies..." | tee -a $LOG_FILE
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt | tee -a $LOG_FILE
else
    echo "❌ requirements.txt not found. Exiting..." | tee -a $LOG_FILE
    exit 1
fi

# 2️⃣ Check if the Fine-Tuned RAG Model Exists
if [ ! -d "models/rag_finetuned" ]; then
    echo "❌ Fine-tuned RAG model not found. Training required." | tee -a $LOG_FILE
    echo "🔄 Running RAG Model Fine-Tuning..." | tee -a $LOG_FILE
    python src/train_rag.py | tee -a $LOG_FILE
else
    echo "✅ Fine-tuned RAG model found. Skipping training..." | tee -a $LOG_FILE
fi

# 3️⃣ Run FAISS Indexing
echo "🔍 Running FAISS Indexing for Knowledge Retrieval..." | tee -a $LOG_FILE
python src/faiss_indexer.py | tee -a $LOG_FILE

if [ $? -ne 0 ]; then
    echo "❌ FAISS Indexing failed. Exiting..." | tee -a $LOG_FILE
    exit 1
fi

echo "✅ FAISS Indexing completed successfully!" | tee -a $LOG_FILE

# 4️⃣ Start the ROS API for Robotic Integration
echo "🤖 Launching ROS API..." | tee -a $LOG_FILE
roslaunch rag_robot_interface start.launch &>> $LOG_FILE &  
ROS_PID=$!
sleep 5  # Give ROS time to initialize

# Check if ROS successfully started
if ps -p $ROS_PID > /dev/null; then
    echo "✅ ROS API started successfully!" | tee -a $LOG_FILE
else
    echo "❌ Failed to start ROS API. Exiting..." | tee -a $LOG_FILE
    exit 1
fi

# 5️⃣ Run Query Testing for RAG-Based Robotics
echo "🧠 Running Query Testing..." | tee -a $LOG_FILE
python src/inference.py | tee -a $LOG_FILE

if [ $? -ne 0 ]; then
    echo "❌ Query Testing failed. Exiting..." | tee -a $LOG_FILE
    exit 1
fi

echo "✅ Query Testing completed successfully!" | tee -a $LOG_FILE

# 6️⃣ Run Evaluation (BLEU Score, Recall@K)
echo "📊 Evaluating Model Performance..." | tee -a $LOG_FILE
python src/evaluation.py | tee -a $LOG_FILE

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed. Exiting..." | tee -a $LOG_FILE
    exit 1
fi

echo "✅ Evaluation completed successfully!" | tee -a $LOG_FILE
echo "🎯 Demo Completed Successfully!" | tee -a $LOG_FILE
