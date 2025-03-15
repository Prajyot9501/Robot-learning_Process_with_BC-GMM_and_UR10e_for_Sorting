import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger  # Import ROS2 service type
from robotiq_gripper_control import RobotiqGripper
from rtde_control import RTDEControlInterface
import time

class GripperNode(Node):
    def __init__(self):
        super().__init__('gripper_node')
        
        # Initialize RTDE Control Interface
        self.rtde_c = RTDEControlInterface("192.168.0.236")  
        self.get_logger().info("RTDE Control Interface connected!")

        # Initialize Robotiq Gripper
        self.gripper = RobotiqGripper(self.rtde_c)
        self.get_logger().info("Robotiq Gripper initialized!")

        # Activate gripper and set initial force/speed
        self.gripper.activate()
        self.gripper.set_force(50)  # Force 0 - 100%
        self.gripper.set_speed(90)  # Speed 0 - 100%
        self.get_logger().info("Gripper activated with force 50% and speed 90%")

        # Define ROS2 services for opening and closing the gripper
        self.srv_open = self.create_service(Trigger, 'open_gripper', self.open_gripper)
        self.srv_close = self.create_service(Trigger, 'close_gripper', self.close_gripper)

    def open_gripper(self, request, response):
        """Service to open the gripper"""
        try:
            self.gripper.open()
            self.get_logger().info("Gripper opened!")
            response.success = True
            response.message = "Gripper is open"
        except Exception as e:
            response.success = False
            response.message = f"Failed to open gripper: {e}"
        return response

    def close_gripper(self, request, response):
        """Service to close the gripper"""
        try:
            self.gripper.close()
            self.get_logger().info("Gripper closed!")
            response.success = True
            response.message = "Gripper is closed"
        except Exception as e:
            response.success = False
            response.message = f"Failed to close gripper: {e}"
        return response

    def shutdown(self):
        """Shutdown gripper and RTDE"""
        self.rtde_c.stopRobot()  # Stop UR control script
        self.get_logger().info("RTDE Control Interface stopped!")
        self.get_logger().info("GripperNode shutting down...")

def main(args=None):
    rclpy.init(args=args)
    node = GripperNode()
    rclpy.spin(node)  # Keep the node running
    node.shutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
