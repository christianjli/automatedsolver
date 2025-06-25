# Solve on Screen through iPhone Mirroring

from pixel_grid_extraction import *
from solver_functions import *
import time
import pyautogui

# Use calibrate.py to find window coordinates
topleft = [5, 80]
bottomright = [338, 790]
window_x = topleft[0]   # x coordinate of mirrored window
window_y = topleft[1]   # y coordinate of mirrored window
window_w = bottomright[0] - topleft[0]   # width of mirrored window
window_h = bottomright[1] - topleft[1]  # height of mirrored window

# Step 1: Take a screenshot
screenshot = pyautogui.screenshot(region=(window_x, window_y, window_w, window_h))
screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# Optional: Save for debugging
cv2.imwrite('latest_screen.png', screen)

# Step 2: Solve puzzle
start_time = time.time()
top_crop, bottom_crop = crop_parameters(screen)
(rows, cols), h_lines, v_lines, cropped = extract_grid_size(screen, top_crop, bottom_crop)
colors = scrape(cropped, False, h_lines, v_lines)
colorcoords = extract_pixel_coords(colors)
color = colorsdict(colorcoords)
matrix = board_matrix(colorcoords, rows, cols)
board = board_matrix_string(matrix)
print(f"Examining Board {cols}x{rows}")
extracted_time = time.time()
print(f'Pixel coordinates and grid size extracted in {extracted_time - start_time:.3f} seconds')

# Run solver
solved_board, paths = pyflow_solver_main(board, color)
solved_time = time.time()
print(f'Puzzle solved in {solved_time - extracted_time:.3f} seconds')

# PyAutoGUI config
crop_top_px = window_h * top_crop
pyautogui.PAUSE = 0
pyautogui.moveTo(100, 200, duration=0.01)
pyautogui.mouseDown()
pyautogui.mouseUp()

# Step 4: Solve through screen mirroring
screen_start = time.time()
for path in paths:
    for i in range(len(path) - 2):
        row1, col1 = path[i]
        row2, col2 = path[i + 1]
        x1, y1 = get_cell_center(row1, col1, h_lines, v_lines, window_x, window_y, crop_top_px)
        x2, y2 = get_cell_center(row2, col2, h_lines, v_lines, window_x, window_y, crop_top_px)

        # Draw with PyAutoGUI
        pyautogui.moveTo(x1, y1, duration=0)
        pyautogui.mouseDown()
        pyautogui.moveTo(x2, y2, duration=0)
        pyautogui.mouseUp()

screen_end = time.time()

print(f'Flows completed in {screen_end - screen_start:.3f} seconds')
print(f'Total time taken: {screen_end - start_time:.3f} seconds')
