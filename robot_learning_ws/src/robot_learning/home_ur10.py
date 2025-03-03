import socket

# UR robot IP and port
UR_IP = "192.168.168.5"
UR_PORT = 30002  # URScript command port

# URScript command to move the robot (Modify the joint values as needed)
COMMAND = "movej([0, -1.57, 0, -1.57, 0, 0], a=1.0, v=0.2)\n" # Move to home position
# COMMAND = "movej([4.48, -0.5, 1.0, -2.5, -1.2, 1.0], a=1.0, v=0.2)\n" # pick and place

# Pick and place
# COMMAND = "movej([4.48, -0.5, 0.8, -2.6, -1.2, 1.0], a=1.0, v=0.2)\n" # 
# COMMAND3 = "movej([4.48, -0.5, 1.0, -2.3, -1.2, 1.0], a=1.0, v=0.2)\n" # go down to pick
# COMMAND4 = "movej([4.48, -0.5, 0.8, -2.6, -1.2, 1.0], a=1.0, v=0.2)\n" # go up
# COMMAND5 = "movej([4.48, -0.5, 1.0, -2.5, -1.2, 1.0], a=1.0, v=0.2)\n" # move to next state

# COMMAND = "set_digital_out(0, True)\n"
# COMMAND = "set_digital_out(0, False)\n"

# Create a socket connection
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((UR_IP, UR_PORT))
    print(f"Connected to UR robot at {UR_IP}:{UR_PORT}")

    # Send command
    # s.send(COMMAND.encode('utf-8'))
    s.send(COMMAND.encode('utf-8'))
    print("Command sent!")

    # Close connection
    s.close()
    print("Connection closed.")

except Exception as e:
    print(f"Error: {e}")