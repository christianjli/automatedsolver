# Calibrate Coordinates of Mirrored Screen for AutoSolver

import cv2
import numpy as np
import pyautogui
import time

# Calibrate mirroring screen coordinates and size, trial and error
window_x = 5   # change this!
window_y = 80   # change this!
window_w = 333   # change this!
window_h = 710  # change this!

print("Starting calibration...")

# Take screenshot of mirrored window
screenshot = pyautogui.screenshot(region=(window_x, window_y, window_w, window_h))
screenshot_np = np.array(screenshot)
image = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

# Show image and ask user to click
clicked_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

# Set up window and mouse callback
cv2.namedWindow("Click Grid Corners")
cv2.setMouseCallback("Click Grid Corners", click_event)

print("Please click the TOP-LEFT of the mirrored screen.")
print("Then click the BOTTOM-RIGHT of the mirrored screen.")
print("Press ESC to leave.")

# Main loop
while True:
    cv2.imshow("Click Grid Corners", image)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or len(clicked_points) >= 2:  # ESC to quit or two clicks done
        break

cv2.destroyAllWindows()

# Print result
if len(clicked_points) >= 2:
    print("Calibration done.")
    print(f"Top-Left: {clicked_points[0]}")
    print(f"Bottom-Right: {clicked_points[1]}")
else:
    print("Calibration not completed.")
