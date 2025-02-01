#!/bin/bash

LOG_DIR="logs"
LOG_FILE="$LOG_DIR/demo_run.log"

mkdir -p $LOG_DIR

echo "🚀 Starting RAG for Robotic Decision-Making Demo (ROS2 Humble) 🚀" | tee $LOG_FILE

# 1️⃣ Install Dependencies
echo "📦 Checking and installing dependencies..." | tee -a $LOG_FILE
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt | tee -a $LOG_FILE
else
    echo "❌ requirements.txt not found. Exiting..." | tee -a $LOG_FILE
    exit 1
fi

# 2️⃣ Source ROS2 Humble Environment
source /opt/ros/humble/setup.bash  # ✅ Updated for Humble

# 3️⃣ Build & Install ROS2 Package (if not built)
echo "🔧 Checking ROS2 package build status..." | tee -a $LOG_FILE
if [ ! -d "install/rag_robot_interface" ]; then
    echo "🔨 Building ROS2 package..." | tee -a $LOG_FILE
    colcon build --packages-select rag_robot_interface | tee -a $LOG_FILE
else
    echo "✅ ROS2 package already built. Skipping build..." | tee -a $LOG_FILE
fi

source install/setup.bash  # ✅ Source ROS2 workspace

# 4️⃣ Check if Fine-Tuned RAG Model Exists
if [ ! -d "models/rag_finetuned" ]; then
    echo "❌ Fine-tuned RAG model not found. Training required." | tee -a $LOG_FILE
    echo "🔄 Running RAG Model Fine-Tuning..." | tee -a $LOG_FILE
    python src/train_rag.py | tee -a $LOG_FILE
else
    echo "✅ Fine-tuned RAG model found. Skipping training..." | tee -a $LOG_FILE
fi

# 5️⃣ Run FAISS Indexing
echo "🔍 Running FAISS Indexing for Knowledge Retrieval..." | tee -a $LOG_FILE
python src/faiss_indexer.py | tee -a $LOG_FILE

if [ $? -ne 0 ]; then
    echo "❌ FAISS Indexing failed. Exiting..." | tee -a $LOG_FILE
    exit 1
fi

echo "✅ FAISS Indexing completed successfully!" | tee -a $LOG_FILE

# 6️⃣ Build Knowledge Graph
echo "📂 Building Knowledge Graph..." | tee -a $LOG_FILE
python src/knowledge_graph.py | tee -a $LOG_FILE

# 7️⃣ Start the ROS2 NLP Node
echo "🤖 Launching ROS2 NLP Node..." | tee -a $LOG_FILE
ros2 launch rag_robot_interface robot_nlp_launch.py &>> $LOG_FILE &
sleep 5  # Give ROS2 time to initialize

# 8️⃣ Verify ROS2 Node is Running
if ros2 node list | grep -q "robot_nlp_interface"; then
    echo "✅ ROS2 NLP Node started successfully!" | tee -a $LOG_FILE
else
    echo "❌ Failed to start ROS2 NLP Node. Exiting..." | tee -a $LOG_FILE
    exit 1
fi

# 9️⃣ Run Query Testing
echo "🧠 Running Query Testing..." | tee -a $LOG_FILE
python src/inference.py | tee -a $LOG_FILE

if [ $? -ne 0 ]; then
    echo "❌ Query Testing failed. Exiting..." | tee -a $LOG_FILE
    exit 1
fi

echo "✅ Query Testing completed successfully!" | tee -a $LOG_FILE

# 🔟 Run Evaluation
echo "📊 Evaluating Model Performance..." | tee -a $LOG_FILE
python src/evaluation.py | tee -a $LOG_FILE

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed. Exiting..." | tee -a $LOG_FILE
    exit 1
fi

echo "✅ Evaluation completed successfully!" | tee -a $LOG_FILE
echo "🎯 Demo Completed Successfully!" | tee -a $LOG_FILE
