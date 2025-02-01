
# RAG for Robotic Decision-Making  

🚀 A **hybrid Retrieval-Augmented Generation (RAG) system** integrating **FAISS-based vector search** and **Knowledge Graphs** to enhance **NLP-driven robotic decision-making** for autonomous systems.

---

## 📌 Overview  

This project enables **UGVs and robotic arms** to:  
✅ **Process natural language queries** related to troubleshooting, task execution, and operational workflows.  
✅ **Retrieve knowledge from robotic manuals, safety protocols, and execution logs** using **FAISS-based similarity search for fast retrieval**.  
✅ **Leverage Knowledge Graphs for structured reasoning**, improving interpretability and inference.  
✅ **Enable real-time task adaptation** through a **ROS2-compatible API**, allowing seamless **human-robot collaboration**.  

---

## 📂 Project Structure  

```plaintext
📂 rag-for-robotic-decision-making/
 ├── 📂 data/
 │    ├── train.json                    # Sample fine-tuning dataset              
 │    ├── robotic_manuals.json          # Robotic manuals & safety protocols
 │    ├── troubleshooting_logs.json     # Logs from real-world robotic failures
 │    ├── ros_commands.json             # Command-to-action mappings
 ├── 📂 models/                          # Pre-trained & fine-tuned RAG models
 │    ├── rag_finetuned/                 # Fine-tuned RAG model
 │    │    ├── config.json
 │    │    ├── pytorch_model.bin
 │    │    ├── tokenizer.json
 │    ├── faiss_index/                    # FAISS vector search index
 │    │    ├── robotic_manuals.faiss
 │    │    ├── troubleshooting_logs.faiss
 │    ├── knowledge_graph/                # Structured knowledge base
 │    │    ├── knowledge_graph.pkl
 ├── 📂 src/                             # Core source code
 │    ├── faiss_indexer.py               # FAISS indexing & retrieval functions
 │    ├── knowledge_graph.py             # Knowledge Graph reasoning
 │    ├── data_loader.py                 # Retrieves documents from FAISS & KG
 │    ├── rag_pipeline.py                # NLP pipeline (RAG + Llama + GPT)
 │    ├── train_rag.py                   # Fine-tuning RAG model
 │    ├── inference.py                    # Runs NLP query inference
 │    ├── evaluation.py                   # Evaluates RAG performance
 ├── 📂 rag_robot_interface/              # ROS2 package for NLP-robot API
 │    ├── __init__.py                     # ROS2 package init file
 │    ├── robot_api.py                    # ROS2 NLP Node
 ├── 📂 launch/                           # ROS2 launch files
 │    ├── robot_nlp_launch.py             # ROS2 launch script
 ├── 📂 notebooks/                        # Jupyter notebooks for testing
 │    ├── rag_pipeline.ipynb              # End-to-end implementation
 ├── 📂 configs/                          # Configuration files
 ├── 📂 logs/                             # Log directory
 │    ├── demo_run.log                    # Output log file
 ├── setup.py                             # ROS2 package setup
 ├── requirements.txt                     # Python dependencies
 ├── demo_run.sh                          # Automated script to run the entire pipeline
 ├── README.md                            # Project overview & instructions
 ├── LICENSE                              # Open-source license
 ├── .gitignore                           # Ignore unnecessary files
```
---

## ⚙️ Installation  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/yourusername/rag-for-robotic-decision-making.git
cd rag-for-robotic-decision-making
```

### **2️⃣ Install Python Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Install ROS2 Dependencies via `rosdep`**  
```bash
# Ensure ROS2 environment is sourced first
source /opt/ros/humble/setup.bash  # Replace 'humble' with your ROS2 version

# Install required ROS2 dependencies
rosdep install --from-paths src --ignore-src -r -y
```

### **4️⃣ Build & Source ROS2 Package**  
```bash
colcon build --packages-select rag_robot_interface
source install/setup.bash
```

### **5️⃣ Verify Installation**  
```bash
python -c "import torch; print(torch.__version__)"  # Check PyTorch installation
python -c "import faiss; print(faiss.__version__)"  # Check FAISS installation
ros2 pkg list | grep rag_robot_interface           # Verify ROS2 package installation
```
---

## 🚀 How to Run the ROS2 NLP Node  
```bash
ros2 launch rag_robot_interface robot_nlp_launch.py
```

---

## 🚀 Running the Demo Script  

To **automate the full pipeline execution**, use the `demo_run.sh` script.

```bash
chmod +x demo_run.sh
./demo_run.sh
```

All output is logged to:  
```bash
cat logs/demo_run.log
```

---

## 🚀 How It Works  

### 1️⃣ **Retrieval-Augmented Generation (RAG) for Robotics**  
🔹 **FAISS-based retrieval** fetches relevant documents from **robotic manuals & execution logs** based on **semantic similarity**.  
🔹 The **Knowledge Graph provides structured reasoning**, helping robots understand workflows, dependencies, and task execution logic.  
🔹 A **fine-tuned RAG model (`facebook/rag-token-nq`, GPT-based embeddings)** integrates retrieved knowledge and generates contextually relevant responses.  
🔹 The **ROS2-compatible API enables real-time decision-making**, allowing robots to process human queries and execute adaptive actions.  

### 2️⃣ **Query Processing Example (Standalone Inference)**  
```python
from src.rag_pipeline import RAGModel

# Initialize the RAG model
model = RAGModel()

# Query the model
query = "How do I recalibrate my robotic arm?"
response = model.answer_query(query)

# Display response
print(f"🤖 Generated Response: {response['response']}")
print(f"📚 Retrieved Documents: {response['retrieved_docs']}")
```
---

## 🚀 Sending a Command to the Robot  

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

rclpy.init()
node = Node("robot_command_publisher")
pub = node.create_publisher(String, "/robot_commands", 10)

msg = String()
msg.data = "Check motor torque levels"
pub.publish(msg)

print("Command sent to robot!")
node.destroy_node()
rclpy.shutdown()
```

---

## 🚨 Troubleshooting Common Issues
| **Issue** | **Fix** |
|-----------|---------|
| `ModuleNotFoundError: No module named 'rclpy'` | Run `pip install rclpy` and ensure ROS2 is sourced (`source /opt/ros/humble/setup.bash`) |
| `ros2 launch not found` | Ensure ROS2 is installed and sourced (`sudo apt install ros-humble-desktop`) |
| `FAISS index file missing` | Run `python src/faiss_indexer.py` before running inference |
| `Robot does not respond` | Ensure the ROS2 NLP node is running (`ros2 node list | grep robot_nlp_interface`) |

---

## 📊 Evaluation Metrics  

The system is evaluated based on:  
✅ **BLEU & ROUGE scores** for text generation quality.  
✅ **Recall@K for retrieval effectiveness**.  
✅ **Response latency for real-time execution**.  

---

## 🔥 Future Improvements  
🚀 **Multi-modal RAG (Images + Text) for vision-based robotic queries**  
🚀 **Enhanced Knowledge Graph for deeper reasoning**  
🚀 **Real-time learning from robot interactions**  

---

## 🤝 Contributing  

Pull requests are welcome! To contribute:  
1. Fork the repo  
2. Create a feature branch: `git checkout -b feature-new-feature`  
3. Commit changes: `git commit -m "Added new feature"`  
4. Push to branch: `git push origin feature-new-feature`  
5. Submit a Pull Request  

---

## 📜 License  

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Star This Repo!  

If you find this useful, please ⭐ the repo and contribute!  
```
