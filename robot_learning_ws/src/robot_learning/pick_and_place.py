import socket
import time
import subprocess
import csv
from datetime import datetime
from rtde_receive import RTDEReceiveInterface  # RTDE for joint states

# UR robot IP and port
UR_IP = "192.168.168.5"
UR_PORT = 30002  # URScript command port

# Define pick and place positions (in radians)
PICK_POSITION = "movej([4.48, -0.5, 1.0, -2.5, -1.2, 1.0], a=0.8, v=0.2)\n"
PLACE_POSITION = "movej([4.2, -0.3, 1.2, -2.0, -1.1, 1.5], a=0.8, v=0.2)\n"

# Initialize RTDE to receive robot joint states
rtde_r = RTDEReceiveInterface(UR_IP)

# File to store collected data
LOG_FILE = "pick_and_place_log.csv"

# Initialize CSV logging
with open(LOG_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "J1", "J2", "J3", "J4", "J5", "J6", "Gripper", "State"])  # Header row

def log_robot_state(gripper_state, operation_state):
    """Logs joint angles, gripper state, and operation status."""
    joint_angles = rtde_r.getActualQ()  # Get joint positions
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp] + joint_angles + [gripper_state, operation_state])

def send_ur_command(command, operation_state):
    """Send a URScript command to the robot and log the state."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((UR_IP, UR_PORT))
        print(f"Connected to UR robot at {UR_IP}:{UR_PORT}")

        s.send(command.encode('utf-8'))
        print(f"Sent command: {command.strip()}")

        # Log robot state while executing the command
        log_robot_state("N/A", operation_state)

        s.close()
        print("Connection closed.")
        time.sleep(2)  # Wait for movement to complete
    except Exception as e:
        print(f"Error sending command: {e}")

def control_gripper(action):
    """Call ROS 2 service to open or close the gripper and log state."""
    cmd = ["ros2", "service", "call", f"/{action}_gripper", "std_srvs/srv/Trigger"]
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Log gripper action
    if "success=True" in result.stdout:
        log_robot_state(action.upper(), "Gripper Action")
    else:
        print(f"Gripper action failed: {result.stdout}")

def pick_and_place():
    """Main function for pick-and-place sequence with logging."""
    print("Moving to pick position...")
    send_ur_command(PICK_POSITION, "Moving to Pick")

    print("Closing gripper to pick the object...")
    control_gripper("close")

    print("Moving to place position...")
    send_ur_command(PLACE_POSITION, "Moving to Place")

    print("Opening gripper to release the object...")
    control_gripper("open")

    print("Pick-and-place operation complete.")

if __name__ == "__main__":
    pick_and_place()
