import cv2
import numpy as np
import json

# Define color ranges in HSV (tweak these for your lighting conditions)
# Lower and upper bounds (H, S, V) for each color
COLOR_RANGES = {
    "yellow": {
        "lower": (20, 100, 100),
        "upper": (30, 255, 255)
    },
    "green": {
        "lower": (40, 50, 50),
        "upper": (75, 255, 255)
    },
    "blue": {
        "lower": (90, 50, 50),
        "upper": (130, 255, 255)
    },
    "red": [
        # Red often spans the hue boundary, so we split it into two ranges
        {
            "lower": (0, 120, 70),
            "upper": (10, 255, 255)
        },
        {
            "lower": (170, 120, 70),
            "upper": (180, 255, 255)
        }
    ]
}

def find_block_center(mask):
    """
    Finds the largest contour in the mask and returns its centroid (cx, cy).
    Returns None if no contour is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def process_frame_for_color(frame, color_name, color_range):
    """
    Thresholds a frame for a given color range, finds the center of the block,
    and returns a shape-(10,) array. 
    We put placeholders (0.0) for extra dimensions.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red is special because it may need two ranges
    if color_name == "red":
        mask_total = None
        for r in color_range:  # color_range is a list of two dicts
            mask = cv2.inRange(hsv, r["lower"], r["upper"])
            mask_total = mask if mask_total is None else mask_total | mask
        mask = mask_total
    else:
        lower = color_range["lower"]
        upper = color_range["upper"]
        mask = cv2.inRange(hsv, lower, upper)

    center = find_block_center(mask)
    if center is None:
        # Return a default (10,) with zeros if not found
        return [0.0]*10
    else:
        cx, cy = center
        # Example format for robosuite shape (10,):
        # [x, y, z, rot_x, rot_y, rot_z, quat_w, quat_x, quat_y, quat_z]
        # For 2D detection, we just fill x,y and put 0.0 in the rest
        return [float(cx), float(cy), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def main():
    cap = cv2.VideoCapture('videos/recording_1.mp4')
    all_frames_data = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Data structure to store this frame's block info
        # e.g., { "frame": index, "yellow": [...], "green": [...], ... }
        frame_data = {"frame": frame_idx}

        # For each color, detect the block and store the (10,) data
        for color_name, color_range in COLOR_RANGES.items():
            block_info = process_frame_for_color(frame, color_name, color_range)
            frame_data[color_name] = block_info

        all_frames_data.append(frame_data)
        frame_idx += 1

    cap.release()

    # Write all data to JSON
    with open("tracked_blocks.json", "w") as f:
        json.dump(all_frames_data, f, indent=2)

if __name__ == '__main__':
    main()
