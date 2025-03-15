import socket
import time
import subprocess
import csv
import json
from datetime import datetime
from rtde_receive import RTDEReceiveInterface

# UR robot IP and port
UR_IP = "192.168.0.236"
UR_PORT = 30002  # URScript command port

# Define positions for pick-and-place operations
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

rtde_r = RTDEReceiveInterface(UR_IP)
LOG_FILE = "pick_and_place_log.json"

def log_robot_state(timestep, color, operation_state, action):
    joint_angles = rtde_r.getActualQ()
    joint_velocities = rtde_r.getActualQd()
    eef_pos = rtde_r.getActualTCPPose()[:3]
    eef_quat = rtde_r.getActualTCPPose()[3:]
    gripper_qpos = [eef_pos[0] * 0.01, -eef_pos[0] * 0.01]  # Simulated gripper position
    gripper_qvel = [val * -2 for val in gripper_qpos]

    data = {
        f"Timestep {timestep}": {
            "Color": color,
            "Observations": {
                "robot0_joint_pos": joint_angles,
                "robot0_joint_vel": joint_velocities,
                "robot0_eef_pos": eef_pos,
                "robot0_eef_quat": eef_quat,
                "robot0_gripper_qpos": gripper_qpos,
                "robot0_gripper_qvel": gripper_qvel
            },
            "Action": action
        }
    }

    with open(LOG_FILE, "a") as file:
        json.dump(data, file, indent=4)


def pick_and_place(color):
    if color not in POSITIONS:
        print(f"Invalid color: {color}")
        return
    positions = POSITIONS[color]
    timestep = 0
    actions = [[0.0, 0.0, 0.0, 0.5, 0.8, 0.2, 1.0]]
    
    for action in actions:
        log_robot_state(timestep, color, "Default Position", action)
        timestep += 1

if __name__ == "__main__":
    for color in POSITIONS.keys():
        pick_and_place(color)