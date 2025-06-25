# Solve given square puzzle image in directory through coordinate and grid size extraction, and SAT solver

from pixel_grid_extraction import *
from solver_functions import *

image1 = cv2.imread('latest_screen.png')
x_crop = 700 / 2796
y_crop = 2030 / 2796

(rows, cols), h_lines, v_lines, cropped = extract_grid_size(image1, x_crop, y_crop)
colors = scrape(cropped, False, h_lines, v_lines)
colorcoords = extract_pixel_coords(colors)
color = colorsdict(colorcoords)
matrix = board_matrix(colorcoords, rows, cols)
board = board_matrix_string(matrix)

print(f"Detected grid size: {rows}x{cols}")
print(f"Detected coordinates at {colorcoords}")
print(f"Colors dictionary is {color}")
print(f"Board matrix is {board}")

solved_board, paths = pyflow_solver_main(board, color)
print(solved_board.split('\n'))
print(paths)
print(solved_board)