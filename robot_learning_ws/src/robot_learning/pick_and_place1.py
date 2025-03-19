import cv2
import numpy as np
import json
import time
import math

COLOR_RANGES = {
    "yellow": {"lower": (20, 100, 100), "upper": (30, 255, 255)},
    "green":  {"lower": (40, 50, 50),  "upper": (75, 255, 255)},
    "blue":   {"lower": (90, 50, 50),  "upper": (130, 255, 255)},
    "red": [
        {"lower": (0, 120, 70),   "upper": (10, 255, 255)},
        {"lower": (170, 120, 70), "upper": (180, 255, 255)}
    ]
}

LOG_FREQUENCY = 10  # in Hz
LOG_PERIOD = 1.0 / LOG_FREQUENCY

def find_block_properties(mask):
    """
    Finds the largest contour in the mask and returns:
      - (cx, cy): the centroid
      - area: contour area
      - angle_deg: angle from minAreaRect (in degrees)
    Returns None if no contour is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    # minAreaRect: ((cx,cy),(w,h), angle)
    rect = cv2.minAreaRect(largest_contour)
    (cx, cy), (w, h), angle_deg = rect

    # Basic centroid from moments
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None

    cx_int = int(M["m10"] / M["m00"])
    cy_int = int(M["m01"] / M["m00"])

    return (cx_int, cy_int, area, angle_deg)

def process_frame_for_color(frame, color_name, color_range, prev_props, dt):
    """
    Thresholds a frame for a given color range, finds block properties,
    and returns a shape-(10,) array: [px, py, pz, qx, qy, qz, qw, vx, vy, vz].
    - px, py, pz: approximate 3D position
    - qx, qy, qz, qw: approximate orientation (only around z-axis)
    - vx, vy, vz: velocity derived from difference in centroid positions / dt
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red can require two ranges
    if color_name == "red":
        mask_total = None
        for r in color_range:
            mask = cv2.inRange(hsv, r["lower"], r["upper"])
            mask_total = mask if mask_total is None else (mask_total | mask)
        mask = mask_total
    else:
        lower = color_range["lower"]
        upper = color_range["upper"]
        mask = cv2.inRange(hsv, lower, upper)

    props = find_block_properties(mask)
    if props is None:
        # Return (10,) zeros if not found
        return [0.0]*10, None

    cx, cy, area, angle_deg = props

    # (1) Approximate 3D position
    #     We'll assume pz is somehow related to area (the bigger the area, the closer the object).
    #     Let K be a scale factor for area->depth, or use a known pinhole model if you know real block size.
    #     For demonstration, we'll do a naive approach: pz = some_factor / sqrt(area).
    #     If area is 0, we fallback to 0.0
    some_factor = 1000.0
    pz = some_factor / (math.sqrt(area) + 1e-5)  # avoid division by zero
    px, py = float(cx), float(cy)

    # (2) Orientation around z:
    #     angle_deg from minAreaRect is typically the rotation in image plane.
    #     Convert angle_deg -> radians
    angle_rad = math.radians(angle_deg)

    # We'll create a quaternion that rotates around z by angle_rad
    # q = [qx, qy, qz, qw]
    qx, qy = 0.0, 0.0
    qz = math.sin(angle_rad / 2.0)
    qw = math.cos(angle_rad / 2.0)

    # (3) Velocity (vx, vy, vz)
    #     If we have a previous centroid, compute difference / dt
    if prev_props is not None:
        (prev_cx, prev_cy, prev_area, prev_angle_deg) = prev_props
        vx = (cx - prev_cx) / dt
        vy = (cy - prev_cy) / dt
        # We can also approximate vz from difference in pz if we want
        prev_pz = some_factor / (math.sqrt(prev_area) + 1e-5)
        vz = (pz - prev_pz) / dt
    else:
        vx, vy, vz = 0.0, 0.0, 0.0

    object_state = [px, py, pz, qx, qy, qz, qw, vx, vy, vz]
    # Return new properties so we can track velocity next time
    new_props = (cx, cy, area, angle_deg)
    return object_state, new_props

def main():
    cap = cv2.VideoCapture('videos/recording_1.mp4')
    all_frames_data = []

    # Track the previous properties for each color
    prev_props = {c: None for c in COLOR_RANGES}
    last_log_time = time.time()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        current_time = time.time()
        if current_time - last_log_time < LOG_PERIOD:
            # Skip frames until we reach desired logging rate
            continue
        last_log_time = current_time

        frame_data = {"frame": frame_idx}

        for color_name, color_range in COLOR_RANGES.items():
            object_state, new_props = process_frame_for_color(
                frame,
                color_name,
                color_range,
                prev_props[color_name],
                LOG_PERIOD
            )
            prev_props[color_name] = new_props
            frame_data[color_name] = object_state

        all_frames_data.append(frame_data)
        frame_idx += 1

    cap.release()

    with open("tracked_blocks.json", "w") as f:
        json.dump(all_frames_data, f, indent=2)

if __name__ == '__main__':
    main()
