import socket
import time
import subprocess
import csv
from datetime import datetime
from rtde_receive import RTDEReceiveInterface  # RTDE for joint states

# UR robot IP and port
UR_IP = "192.168.168.5"
UR_PORT = 30002  # URScript command port

# Define pick and place positions (in radians) for Yellow Color
DEFAULT_POSITION_Yellow = "movej([4.35, -0.95, 1.5, -2.1, -1.5, 1.5], a=0.8, v=0.2)\n"

PICK_POSITION = "movej([4.35, -0.8, 1.50, -2.1, -1.5, 1.5], a=0.8, v=0.2)\n"
UP_POSITION = "movej([4.35, -0.98, 1.5, -2.1, -1.5, 1.5], a=0.8, v=0.2)\n"
MOVE_POSITION = "movej([4.95, -0.95, 1.5, -2.4, -1.2, 1.7], a=0.8, v=0.2)\n"
PLACE_POSITION = "movej([4.95, -0.74, 1.5, -2.6, -1.2, 1.7], a=0.8, v=0.2)\n"
UP_POSITION_1 = "movej([4.95, -0.95, 1.5, -2.6, -1.2, 1.7], a=0.8, v=0.2)\n"


# Define pick and place positions (in radians) for Green Color
DEFAULT_POSITION_GREEN = "movej([4.24, -0.95, 1.5, -2.1, -1.5, 1.2], a=0.8, v=0.2)\n"

PICK_POSITION_GREEN = "movej([4.24, -0.8, 1.5, -2.1, -1.5, 1.2], a=0.8, v=0.2)\n"
UP_POSITION_GREEN = "movej([4.24, -0.95, 1.5, -2.1, -1.5, 1.2], a=0.8, v=0.2)\n"
MOVE_POSITION_GREEN = "movej([4.70, -0.95, 1.5, -2.1, -1.2, 1.5], a=0.8, v=0.2)\n"
PLACE_POSITION_GREEN = "movej([4.70, -0.8, 1.5, -2.1, -1.2, 1.5], a=0.8, v=0.2)\n"
UP_POSITION_1_GREEN = "movej([4.70, -0.95, 1.5, -2.1, -1.2, 1.5], a=0.8, v=0.2)\n"


# Define pick and place positions (in radians) for Blue Color
DEFAULT_POSITION_BLUE = "movej([4.30, -0.8, 1.2, -2.3, -1.0, 1.0], a=0.8, v=0.2)\n"

PICK_POSITION_BLUE = "movej([4.30, -0.56, 1.2, -2.3, -1.0, 1.0], a=0.8, v=0.2)\n"
UP_POSITION_BLUE = "movej([4.30, -0.90, 1.2, -2.3, -1.0, 1.0], a=0.8, v=0.2)\n"
MOVE_POSITION_BLUE = "movej([4.95, -0.52, 0.8, -2.5, -1.4, 1.6], a=0.8, v=0.2)\n"
PLACE_POSITION_BLUE = "movej([4.95, -0.37, 0.8, -2.5, -1.4, 1.6], a=0.8, v=0.2)\n"
UP_POSITION_1_BLUE = "movej([4.95, -0.52, 0.8, -2.5, -1.4, 1.6], a=0.8, v=0.2)\n"

# Define pick and place positions (in radians) for Red Color
DEFAULT_POSITION_RED = "movej([4.30, -0.52, 0.8, -2.5, -1.4, 1.6], a=0.8, v=0.2)\n"

PICK_POSITION_RED = "movej([4.30, -0.40, 0.8, -2.1, -1.4, 1.2], a=0.8, v=0.2)\n"
UP_POSITION_RED = "movej([4.30, -0.55, 0.8, -2.5, -1.4, 1.6], a=0.8, v=0.2)\n"
MOVE_POSITION_RED = "movej([4.67, -0.55, 0.8, -2.5, -1.4, 1.6], a=0.8, v=0.2)\n"
PLACE_POSITION_RED = "movej([4.67, -0.39, 0.8, -2.1, -1.2, 1.2], a=0.8, v=0.2)\n"
UP_POSITION_1_RED = "movej([4.67, -0.55, 0.8, -2.5, -1.4, 1.6], a=0.8, v=0.2)\n"

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

def pick_and_place_yellow():
    """Main function for pick-and-place sequence with logging."""

    send_ur_command(DEFAULT_POSITION_Yellow, "Default")
    
    print("Moving to pick position...")
    send_ur_command(PICK_POSITION, "Moving to Pick")

    print("Closing gripper to pick the object...")
    control_gripper("close")

    print("Moving to place position...")
    send_ur_command(UP_POSITION, "Moving to UP")
    send_ur_command(MOVE_POSITION, "Moving")
    send_ur_command(PLACE_POSITION, "Moving to Down")

    print("Opening gripper to release the object...")
    control_gripper("open")

    send_ur_command(UP_POSITION_1, "Release Default")

    print("Pick-and-place operation complete.")

def pick_and_place_green():
    """Main function for pick-and-place sequence with logging."""

    send_ur_command(DEFAULT_POSITION_GREEN, "Default")
    
    print("Moving to pick position...")
    send_ur_command(PICK_POSITION_GREEN, "Moving to Pick")

    print("Closing gripper to pick the object...")
    control_gripper("close")

    print("Moving to place position...")
    send_ur_command(UP_POSITION_GREEN, "Moving to UP")
    send_ur_command(MOVE_POSITION_GREEN, "Moving")
    send_ur_command(PLACE_POSITION_GREEN, "Moving to Down")

    print("Opening gripper to release the object...")
    control_gripper("open")

    send_ur_command(UP_POSITION_1_GREEN, "Release Default")

    print("Pick-and-place operation complete.")

def pick_and_place_blue():
    """Main function for pick-and-place sequence with logging."""

    send_ur_command(DEFAULT_POSITION_BLUE, "Default")
    
    print("Moving to pick position...")
    send_ur_command(PICK_POSITION_BLUE, "Moving to Pick")

    print("Closing gripper to pick the object...")
    control_gripper("close")

    print("Moving to place position...")
    send_ur_command(UP_POSITION_BLUE, "Moving to UP")
    send_ur_command(MOVE_POSITION_BLUE, "Moving")
    send_ur_command(PLACE_POSITION_BLUE, "Moving to Down")

    print("Opening gripper to release the object...")
    control_gripper("open")

    send_ur_command(UP_POSITION_1_BLUE, "Release Default")

    print("Pick-and-place operation complete.")

def pick_and_place_red():
    """Main function for pick-and-place sequence with logging."""

    send_ur_command(DEFAULT_POSITION_RED, "Default")
    
    print("Moving to pick position...")
    send_ur_command(PICK_POSITION_RED, "Moving to Pick")

    print("Closing gripper to pick the object...")
    control_gripper("close")

    print("Moving to place position...")
    send_ur_command(UP_POSITION_RED, "Moving to UP")
    send_ur_command(MOVE_POSITION_RED, "Moving")
    send_ur_command(PLACE_POSITION_RED, "Moving to Down")

    print("Opening gripper to release the object...")
    control_gripper("open")

    send_ur_command(UP_POSITION_1_RED, "Release Default")

    print("Pick-and-place operation complete.")

if __name__ == "__main__":
    pick_and_place_yellow()
    pick_and_place_green()
    pick_and_place_blue()
    pick_and_place_red()
