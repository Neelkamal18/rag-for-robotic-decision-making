import rclpy
from rag_robot_interface.robot_api import RobotNLPInterface

def run_query():
    """Runs a test query using the RAG-based NLP system in ROS2."""
    rclpy.init()  #Initialize ROS2 before creating the node

    try:
        node = RobotNLPInterface()
        query = input("Enter your query: ")  # Allow user-defined queries
        
        print(f"📥 Query: {query}")
        response = node.rag_model.answer_query(query)
        
        print(f"📢 Generated Response: {response['response']}")
        print(f"📚 Retrieved Documents: {response['retrieved_docs']}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    finally:
        node.destroy_node()  # Properly clean up the node
        rclpy.shutdown()  # Shutdown ROS2

if __name__ == "__main__":
    run_query()
