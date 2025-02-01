# 🤖 RAG for Robotic Decision-Making  

🚀 A **hybrid Retrieval-Augmented Generation (RAG) system** integrating **FAISS-based vector search** and **Knowledge Graphs** to enhance **NLP-driven robotic decision-making** for autonomous systems.

---

## 📌 Overview  

This project enables **UGVs and robotic arms** to:  
✅ **Process natural language queries** related to troubleshooting, task execution, and operational workflows.  
✅ **Retrieve knowledge from robotic manuals, safety protocols, and execution logs** using **FAISS for fast similarity search**.  
✅ **Leverage Knowledge Graphs for structured reasoning**, improving interpretability and inference.  
✅ **Enable real-time task adaptation** through a **ROS-compatible API**, allowing seamless **human-robot collaboration**.  

---

## 📂 Project Structure  

```plaintext
📂 rag-for-robotic-decision-making/
 ├── 📂 data/                    
 │    ├── robotic_manuals.json          # Robotic manuals & safety protocols
 │    ├── troubleshooting_logs.json     # Logs from real-world robotic failures
 │    ├── ros_commands.json             # Command-to-action mappings
 │    ├── embeddings/                    # FAISS index storage
 │    │    ├── robotic_manuals.faiss
 │    │    ├── troubleshooting_logs.faiss
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
 │    ├── robot_api.py                   # ROS-compatible API
 │    ├── inference.py                    # Runs NLP query inference
 │    ├── evaluation.py                   # Evaluates RAG performance
 ├── 📂 notebooks/                        # Jupyter notebooks for testing
 ├── 📂 configs/                          # Configuration files
 ├── requirements.txt                     # Python dependencies
 ├── demo_run.sh                         # Automated script to run the entire pipeline
 ├── logs/                               # 📂 Log directory
 │    ├── demo_run.log                   # Output log file
 ├── README.md                           # Project overview & instructions
 ├── LICENSE                             # Open-source license
 ├── .gitignore                          # Ignore unnecessary files
```

## ⚙️ Installation  

Clone the repository:  
```bash
git clone https://github.com/yourusername/rag-for-robotic-decision-making.git
cd rag-for-robotic-decision-making
```

Install dependencies:  
```bash
pip install -r requirements.txt
```

### ✅ **Yes, this section is correct for your README!**  
However, I have **slightly refined it** for better readability and clarity while keeping the same meaning.  

--- 

## 🚀 How to Run the Demo Script  

To **automate the full pipeline execution**, use the provided `demo_run.sh` script.  

### 1️⃣ **Give Execution Permission:**  
```bash
chmod +x demo_run.sh
```

### 2️⃣ **Run the Script:**  
```bash
./demo_run.sh
```

### 3️⃣ **Check Logs:**  
All output is logged to:  
```bash
cat logs/demo_run.log
```

### 📌 **What This Script Does:**  
✅ **Installs dependencies** (if not already installed)  
✅ **Runs FAISS Indexing** to prepare knowledge retrieval  
✅ **Starts the ROS API** for robotic integration  
✅ **Runs inference** for RAG-based robotic query resolution  
✅ **Evaluates model performance** using **BLEU Score & Recall@K**  
✅ **Handles errors gracefully** and **logs all actions**  

---

## 🚀 How It Works  

### 1️⃣ **Retrieval-Augmented Generation (RAG) for Robotics**  
🔹 FAISS retrieves relevant documents from **robotic manuals & execution logs** based on **semantic similarity**.  
🔹 The **Knowledge Graph provides structured reasoning**, helping robots understand workflows and dependencies.  
🔹 A **fine-tuned RAG model (`facebook/rag-token-nq`, GPT-based embeddings)** generates contextual responses.  

### 2️⃣ **Query Processing Example**  
```python
from src.rag_pipeline import RAGModel

model = RAGModel()
response = model.answer_query("How do I recalibrate my robotic arm?")
print(response)
```

### 3️⃣ **ROS-Compatible API for Robotic Integration**  
Start the robot NLP interface:  
```bash
roslaunch rag_robot_interface start.launch
```

Send a command to the robot:  
```python
import rospy
from std_msgs.msg import String

rospy.init_node("robot_command")
pub = rospy.Publisher("/robot_commands", String, queue_size=10)
pub.publish("Check motor torque levels")
```

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
