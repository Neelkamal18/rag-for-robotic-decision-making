#!/bin/bash

echo "🚀 Starting RAG for Robotic Decision-Making Demo 🚀"

# 1️⃣ Install Dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# 2️⃣ Start the ROS API for Robotic Integration
echo "🤖 Launching ROS API..."
roslaunch rag_robot_interface start.launch &
sleep 5  # Give ROS time to initialize

# 3️⃣ Run Inference for RAG-Based Robotics Querying
echo "🧠 Running RAG Inference..."
python src/inference.py

# 4️⃣ Run Evaluation (BLEU Score, Recall@K)
echo "📊 Evaluating Model Performance..."
python src/evaluation.py

echo "🎯 Demo Completed Successfully!"
