import rospy
from std_msgs.msg import String
from rag_pipeline import RAGModel

class RobotNLPInterface:
    def __init__(self):
        rospy.init_node("robot_nlp_interface")
        self.rag_model = RAGModel()
        self.sub = rospy.Subscriber("/robot_commands", String, self.process_command)
        self.pub = rospy.Publisher("/robot_response", String, queue_size=10)
    
    def process_command(self, msg):
        """Processes the command and generates a response using RAG."""
        response = self.rag_model.answer_query(msg.data)
        self.pub.publish(response)

if __name__ == "__main__":
    RobotNLPInterface()
    rospy.spin()
