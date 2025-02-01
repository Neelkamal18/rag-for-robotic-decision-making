import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from src.rag_pipeline import RAGModel

class RobotNLPInterface(Node):
    def __init__(self):
        super().__init__('robot_nlp_interface')

        # Declare ROS2 parameters
        self.declare_parameter('command_topic', '/robot_commands')
        self.declare_parameter('response_topic', '/robot_response')

        # Get parameters (allows dynamic configuration)
        command_topic = self.get_parameter('command_topic').get_parameter_value().string_value
        response_topic = self.get_parameter('response_topic').get_parameter_value().string_value

        # Initialize RAG model
        try:
            self.rag_model = RAGModel()
            self.get_logger().info("RAG Model Loaded Successfully")
        except Exception as e:
            self.get_logger().error(f"Error loading RAG Model: {str(e)}")
            self.rag_model = None

        # ROS2 Publisher & Subscriber
        self.subscriber = self.create_subscription(String, command_topic, self.process_command, 10)
        self.publisher = self.create_publisher(String, response_topic, 10)

        self.get_logger().info(f"Robot NLP Interface Node Started - Listening on {command_topic}")

    def process_command(self, msg):
        """Processes the command and generates a response using RAG."""
        self.get_logger().info(f"Received command: {msg.data}")

        if not self.rag_model:
            self.get_logger().error("RAG Model is not available.")
            return

        response_text = self.rag_model.answer_query(msg.data)["response"]
        
        response_msg = String()
        response_msg.data = response_text
        self.publisher.publish(response_msg)

        self.get_logger().info(f"Sent response: {response_text}")

def main(args=None):
    rclpy.init(args=args)
    node = RobotNLPInterface()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown signal received. Exiting...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
