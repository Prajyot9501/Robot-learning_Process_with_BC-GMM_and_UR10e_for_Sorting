import socket

robot_ip = "192.168.168.5"
port = 30002  # URScript control interface

# URScript command to close the gripper
close_gripper_script = "socket_open('192.168.168.83', 50002)\n socket_send_string('rq_close()')\n socket_close()\n"

# Connect to the UR controller
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((robot_ip, port))

# Send URScript command
s.send(close_gripper_script.encode('utf-8'))
s.close()
