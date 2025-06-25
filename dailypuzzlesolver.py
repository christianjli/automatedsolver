from autosolver_functions import *
import time
from pynput import keyboard

# Use calibrate.py to find window coordinates
topleft = [5, 80]
bottomright = [338, 790]

# Get window coordinates
window_x, window_y, window_w, window_h = get_window_rect(topleft, bottomright)

puzzle_count = 0
puzzle_start = time.time()
extraction_times = []
solve_times = []

exception_start_time = None
esc_prompt_shown = False
should_exit = False

# Define ESC key handler
def on_press(key):
    global should_exit
    if key == keyboard.Key.esc:
        should_exit = True
        print("\nQuitting Program")

# Start the keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

while not should_exit:
    try:
        # Take screenshot
        screen = take_screenshot(window_x, window_y, window_w, window_h)

        # Extract board coordinates and grid size
        start_time = time.time()
        top_crop, bottom_crop = crop_parameters(screen)
        (rows, cols), h_lines, v_lines, cropped = extract_grid_size(screen, top_crop, bottom_crop)
        colors = scrape(cropped, False, h_lines, v_lines)
        colorcoords = extract_pixel_coords(colors) # tower pack 132, 9x9 level 9 from classic
        color = colorsdict(colorcoords)
        matrix = board_matrix(colorcoords, rows, cols)
        board = board_matrix_string(matrix)
        extract_time = time.time() - start_time
        print(f"\nExamining Board {cols}x{rows} (Puzzle {puzzle_count + 1})")

        # Solve board
        paths = solve_board(board, color, debug=False)
        puzzle_count += 1

        # Draw paths
        crop_top_px = window_h * top_crop
        draw_paths(paths, h_lines, v_lines, window_x, window_y, crop_top_px, debug=False)
        solve_time = time.time() - start_time
        print(f'Puzzle {puzzle_count} solved in {solve_time:.3f} seconds')

        # Next level, reserve downtime for clicking and pixel extraction
        time.sleep(0.2)
        pyautogui.moveTo((topleft[0] + bottomright[0]) / 2, (topleft[1] + bottomright[1]) / 2, duration=0)
        time.sleep(0.3)
        pyautogui.click()
        time.sleep(0.5)
        
        extraction_times.append(extract_time)
        solve_times.append(solve_time)

        # Reset stuck tracker when successful run starts
        exception_start_time = None
        esc_prompt_shown = False
    
    except Exception as e:
        # If first time in exception, track the time
        if exception_start_time is None:
            exception_start_time = time.time()
        # If we've been in exceptions for 5+ seconds and haven't shown message
        if not esc_prompt_shown and (time.time() - exception_start_time) >= 5:
            print("\n(Press ESC to quit the program...)\n")
            esc_prompt_shown = True

        # Sleep to allow manual handling (ads or home screen)
        time.sleep(1)
        continue

listener.stop()
if len(extraction_times) != 0 and len(solve_times) != 0:
    avg_extraction = sum(extraction_times) / len(extraction_times)
    avg_solve = sum(solve_times) / len(solve_times)
    print(f'\nSolved {puzzle_count} puzzles in {time.time() - puzzle_start:.3f} seconds')
    print(f'Average extraction time: {avg_extraction:.3f}')
    print(f'Average solve time: {avg_solve:.3f}')