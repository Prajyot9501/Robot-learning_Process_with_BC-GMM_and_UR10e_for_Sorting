import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import socket
import time
import subprocess
import json
import math
from datetime import datetime

from scipy.spatial.transform import Rotation as R
from rtde_receive import RTDEReceiveInterface

# UR robot IP and port
UR_IP = "192.168.168.5"
UR_PORT = 30002
LOG_FILE = "pick_and_place_log_6.json"

# Positions for pick-and-place
POSITIONS = {
    "yellow": {
        "default": [4.35, -0.95, 1.5, -2.1, -1.5, 1.5],
        "pick": [4.35, -0.8, 1.5, -2.1, -1.5, 1.5],
        "up": [4.35, -0.98, 1.5, -2.1, -1.5, 1.5],
        "move": [4.95, -0.95, 1.5, -2.4, -1.2, 1.7],
        "place": [4.95, -0.74, 1.5, -2.6, -1.2, 1.7],
        "up_final": [4.95, -0.95, 1.5, -2.6, -1.2, 1.7]
    },
    "green": {
        "default": [4.24, -0.95, 1.5, -2.1, -1.5, 1.2],
        "pick": [4.24, -0.8, 1.5, -2.1, -1.5, 1.2],
        "up": [4.24, -0.95, 1.5, -2.1, -1.5, 1.2],
        "move": [4.70, -0.95, 1.5, -2.1, -1.2, 1.5],
        "place": [4.70, -0.8, 1.5, -2.1, -1.2, 1.5],
        "up_final": [4.70, -0.95, 1.5, -2.1, -1.2, 1.5]
    },
    "blue": {
        "default": [4.30, -0.8, 1.2, -2.3, -1.0, 1.0],
        "pick": [4.30, -0.56, 1.2, -2.3, -1.0, 1.0],
        "up": [4.30, -0.90, 1.2, -2.3, -1.0, 1.0],
        "move": [4.95, -0.52, 0.8, -2.5, -1.4, 1.6],
        "place": [4.95, -0.37, 0.8, -2.5, -1.4, 1.6],
        "up_final": [4.95, -0.52, 0.8, -2.5, -1.4, 1.6]
    },
    "red": {
        "default": [4.30, -0.52, 0.8, -2.5, -1.4, 1.6],
        "pick": [4.30, -0.40, 0.8, -2.1, -1.4, 1.2],
        "up": [4.30, -0.55, 0.8, -2.5, -1.4, 1.6],
        "move": [4.67, -0.55, 0.8, -2.5, -1.4, 1.6],
        "place": [4.67, -0.39, 0.8, -2.1, -1.2, 1.2],
        "up_final": [4.67, -0.55, 0.8, -2.5, -1.4, 1.6]
    }
}

# Logging frequency for the trajectory
LOG_FREQUENCY = 10  # 10 Hz
LOG_PERIOD = 1.0 / LOG_FREQUENCY

GLOBAL_TIMESTEP = 0


class PickPlaceNode(Node):
    """
    ROS 2 node that:
      1) Subscribes to Robotiq 2F-85 joint states.
      2) Receives UR states via RTDE.
      3) Logs each step with real gripper states.
    """

    def __init__(self):
        super().__init__("pick_place_node")

        # 1) Subscribe to gripper joint states
        self.subscription = self.create_subscription(
            JointState,
            "/robotiq_2f85/joint_states",  # Adjust if your driver uses a different topic
            self.gripper_callback,
            10
        )
        # Store latest gripper pos, vel
        self.gripper_qpos = [0.0, 0.0]
        self.gripper_qvel = [0.0, 0.0]

        # 2) RTDE to read UR states
        self.rtde_r = RTDEReceiveInterface(UR_IP)
        
        # We do not spin automatically here – user can call self.run_pick_and_place()

    def gripper_callback(self, msg: JointState):
        """
        Called whenever we get a new JointState from the Robotiq driver.
        Typically the 2F-85 publishes 1 or 2 joint states (fingerA, fingerB).
        """
        if len(msg.position) >= 2 and len(msg.velocity) >= 2:
            self.gripper_qpos = [msg.position[0], msg.position[1]]
            self.gripper_qvel = [msg.velocity[0], msg.velocity[1]]

    def log_robot_state(self, step_count: int, color: str, operation_state: str):
        """
        Logs UR states plus real gripper_qpos, gripper_qvel from our subscriber.
        """

        # Gather UR data
        joint_angles_full = self.rtde_r.getActualQ()       # 6 joints
        joint_velocities_full = self.rtde_r.getActualQd()  # 6 joint velocities
        tcp_pose = self.rtde_r.getActualTCPPose()          # [x, y, z, rx, ry, rz]
        tcp_speed = self.rtde_r.getActualTCPSpeed()        # [vx, vy, vz, wx, wy, wz]

        x, y, z = tcp_pose[:3]
        rx, ry, rz = tcp_pose[3:]
        rot = R.from_rotvec([rx, ry, rz])
        qx, qy, qz, qw = rot.as_quat()

        vx, vy, vz = tcp_speed[:3]
        wx, wy, wz = tcp_speed[3:]

        # Use first 3 joints if you want shape (3,)
        joint_pos_3 = joint_angles_full[:3]
        joint_vel_3 = joint_velocities_full[:3]

        joint_pos_cos_3 = [math.cos(j) for j in joint_pos_3]
        joint_pos_sin_3 = [math.sin(j) for j in joint_pos_3]

        # Instead of placeholders, read from self.gripper_qpos/vel
        objects = [0.0]*10  # placeholder for your detection
        action = [x, y, z, qx, qy, qz, qw]

        data = {
            f"Timestep {step_count}": {
                "Color": color,
                "Operation": operation_state,
                "Observations": {
                    "object": objects,
                    "robot0_eef_pos": [x, y, z],
                    "robot0_eef_quat": [qx, qy, qz, qw],
                    "robot0_eef_vel_ang": [wx, wy, wz],
                    "robot0_eef_vel_lin": [vx, vy, vz],
                    "robot0_gripper_qpos": self.gripper_qpos,
                    "robot0_gripper_qvel": self.gripper_qvel,
                    "robot0_joint_pos": joint_pos_3,
                    "robot0_joint_pos_cos": joint_pos_cos_3,
                    "robot0_joint_pos_sin": joint_pos_sin_3,
                    "robot0_joint_vel": joint_vel_3
                },
                "Action": action
            }
        }

        with open(LOG_FILE, "a") as file:
            json.dump(data, file, indent=4)

    def send_ur_command_and_log(self, joint_positions, color, operation_state, move_time=2.0):
        """
        Sends move command & logs at LOG_FREQUENCY for 'move_time' seconds.
        """
        global GLOBAL_TIMESTEP

        command = f"movej({joint_positions}, a=0.7, v=0.2)\n"
        start = time.time()
        try:
            with socket.create_connection((UR_IP, UR_PORT)) as s:
                s.sendall(command.encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f"Error sending command: {e}")
            return

        while True:
            elapsed = time.time() - start
            # Log this step
            self.log_robot_state(GLOBAL_TIMESTEP, color, operation_state)
            GLOBAL_TIMESTEP += 1

            if elapsed >= move_time:
                break
            time.sleep(LOG_PERIOD)

    def control_gripper(self, action_str):
        cmd = [
            "ros2", "service", "call",
            f"/{action_str}_gripper",
            "std_srvs/srv/Trigger"
        ]
        self.get_logger().info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if "success: True" in result.stdout:
            self.get_logger().info(f"Gripper {action_str} successful.")
        else:
            self.get_logger().error(f"Gripper {action_str} failed: {result.stdout}")

    def pick_and_place(self, color: str):
        if color not in POSITIONS:
            self.get_logger().error(f"Invalid color: {color}")
            return
        positions = POSITIONS[color]

        # Move to default
        self.send_ur_command_and_log(positions["default"], color, "Default", move_time=2.0)
        # Move to pick, close gripper
        self.send_ur_command_and_log(positions["pick"], color, "Pick", move_time=2.0)
        self.control_gripper("close")
        # Move up + move
        self.send_ur_command_and_log(positions["up"], color, "Up", move_time=2.0)
        self.send_ur_command_and_log(positions["move"], color, "Move", move_time=2.0)
        # Place + open
        self.send_ur_command_and_log(positions["place"], color, "Place", move_time=2.0)
        self.control_gripper("open")
        # Up final
        self.send_ur_command_and_log(positions["up_final"], color, "UpFinal", move_time=2.0)

        self.get_logger().info(f"Pick-and-place operation for {color} complete.")

    def run_pick_and_place(self):
        """
        Example: run the pick-and-place for each color.
        You can adapt this or call it from outside.
        """
        for c in POSITIONS.keys():
            self.pick_and_place(c)


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceNode()
    # Option A: Spin in separate thread while you do picking:
    # rclpy.spin(node)
    # Option B: Just do picking here:
    node.run_pick_and_place()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
