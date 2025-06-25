import cv2
from pixel_grid_extraction import *
from solver_functions import *
import pyautogui
import numpy as np
import time

def get_window_rect(topleft, bottomright):
    '''Get window coordinates.'''
    x, y = topleft
    w = bottomright[0] - x
    h = bottomright[1] - y
    return x, y, w, h

def take_screenshot(x, y, w, h, filename='latest_screen.png'):
    '''Take screenshot for image processing.'''
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, screen)
    return screen

def solve_board(board, color, debug=False):
    '''Solve board through solver functions.'''
    solved_board, paths = pyflow_solver_main(board, color)
    if debug:
        print("\nSolved Board:\n", solved_board)
        print("\nPaths:")
        for path in paths:
            print(path)
    return paths

def draw_paths(paths, h_lines, v_lines, window_x, window_y, crop_top_px, debug=False):
    '''Use iPhone mirroring to draw paths, completing the puzzle.'''
    pyautogui.PAUSE = 0
    # "Wake up" mouse or clear state (optional)
    pyautogui.moveTo(100, 200, duration=0.01)
    pyautogui.click()

    screen_start = time.time()

    for path in paths:
        if debug:
            print("Drawing path:", path)

        for i in range(len(path) - 2):
            row1, col1 = path[i]
            row2, col2 = path[i + 1]

            x1, y1 = get_cell_center(row1, col1, h_lines, v_lines, window_x, window_y, crop_top_px)
            x2, y2 = get_cell_center(row2, col2, h_lines, v_lines, window_x, window_y, crop_top_px)

            pyautogui.moveTo(x1, y1, duration=0)
            pyautogui.mouseDown()
            pyautogui.moveTo(x2, y2, duration=0)
            pyautogui.mouseUp()

    screen_end = time.time()
    print(f'Flows completed in {screen_end - screen_start:.3f} seconds')
