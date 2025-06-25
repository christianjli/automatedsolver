import cv2
import numpy as np
from sklearn.cluster import KMeans
import random
from collections import deque
from itertools import permutations

# Remove near-duplicate lines with a stricter margin
def remove_close_lines(lines, threshold=15):
    filtered = []
    for l in sorted(lines):
        if not filtered or abs(filtered[-1] - l) > threshold:
            filtered.append(l)
    return filtered

def crop_parameters(image, bottom_limit_px=570):
    image_h = image.shape[0]

    # Scan for content "start" near top 20%
    scan_top = int(0.1 * image_h)
    scan_bottom = int(0.5 * image_h)
    gray = cv2.cvtColor(image[scan_top:scan_bottom], cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Get top-most y for cropping
    if lines is not None:
        y_coords = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            y_coords.extend([y1, y2])
        y_top = min(y_coords) + scan_top
    else:
        y_top = scan_top

    # Force bottom limit
    y_bottom = min(image_h, bottom_limit_px)

    # Normalize as fraction
    x_crop = y_top / image_h
    y_crop = y_bottom / image_h

    return x_crop, y_crop
    
def extract_grid_size(image1, x_crop, y_crop):
  cropped = image1[int((x_crop) * len(image1)) : int((y_crop) * len(image1))]
  gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  kernel = np.ones((3, 3), np.uint8)
  binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)  # Fill gaps
  binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)  # Expand circles

  # Apply mask to remove noise
  cleaned = cv2.bitwise_and(cropped, cropped, mask=binary_mask)
  restored = cv2.bitwise_or(cropped, cropped, mask=binary_mask)
  hsv = cv2.cvtColor(restored, cv2.COLOR_BGR2HSV)

  # Step 2: Detect Grid Lines
  edges = cv2.Canny(blurred, 50, 150)
  lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=100, maxLineGap=10)

  # Identify grid size based on detected lines
  h_lines, v_lines = [], []
  for line in lines:
      x1, y1, x2, y2 = line[0]
      if abs(x1 - x2) < 5:  # Vertical line threshold
          v_lines.append(x1)
      if abs(y1 - y2) < 5:  # Horizontal line threshold
          h_lines.append(y1)

  v_lines = remove_close_lines(v_lines)
  h_lines = remove_close_lines(h_lines)

  grid_size = (len(h_lines) - 1, len(v_lines) - 1)

  return grid_size, h_lines, v_lines, cropped

def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def get_dominant_color(image, k=1, image_processing_size = None):
    #resize image if new dims provided
    if image_processing_size:
        image = cv2.resize(image, image_processing_size, interpolation = cv2.INTER_AREA)
    #reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    #run kmeans
    km = KMeans(n_clusters = k, n_init = 10)
    km.fit(image)
    #extract dom color
    dominant_color = km.cluster_centers_[0]
    dominant_color = dominant_color.astype(int)
    return dominant_color

def scrape(image, flag, h_lines, v_lines):
  # blur = cv2.GaussianBlur(image[0:-1,], (5, 5), 0)
  blur = cv2.GaussianBlur(image[0:-1,], (7, 7), 0) # don't blur too much
  cann = cv2.Canny(blur, 100, 280)
  # cv2_imshow(cann)

  (cnts, hierarchy) = cv2.findContours(cann.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  coins = image.copy()
  cv2.drawContours(coins, cnts, -1, (255, 255, 255), 1)
  fixedcoins = fixColor(coins).copy()
  coords = []
  for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(coins, (x,y), (x+w, y+h), (255, 255, 255), 2)
    coord = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
    coords.append(coord)

  color_groups = []

  for coord in coords:
    yi, yf, xi, xf = coord[0][1], coord[2][1], coord[0][0], coord[2][0]
    segment = coins[yi: yf, xi: xf]
    dom = get_dominant_color(segment, k=1) # gets dominant color
    w, h = (xf - xi), (yf - yi) # width and height
    x, y = xi, yi # x and y coords
    # print(dom)
    flag and cv2.imshow("segment", segment) # show segment
    flag and print(x, y, w, h)

    row = min(range(len(h_lines)), key=lambda i: abs(h_lines[i] - y))
    col = min(range(len(v_lines)), key=lambda i: abs(v_lines[i] - x))
    flag and print(row, col)

    # too small or too big
    minheight = len(image) * 0.035 # originally 0.01
    minwidth = len(image[0]) * 0.045 # originally 0.01
    if w < minwidth or h < minheight: # don't classify if bounding box too small
      flag and print("Too Small")
      continue
    maxheight = len(image) * 0.9
    maxwidth = len(image[0]) * 0.9
    # print(f'max: {maxwidth}, {maxheight}') # reference
    if w > maxwidth or h > maxheight: # takes up 90% of the image in width or height
      flag and print("Too Big")
      continue

    flag and print(dom[0], dom[1], dom[2])

    # find color groups
    color_groups.append((row, col, (dom[0], dom[1], dom[2])))

  return color_groups

def extract_pixel_coords(color_groups):
  colors = []
  for color in color_groups:
    row, col, (d0, d1, d2) = color
    found_group = False
    if colors == []:
      colors.append([color])
    else:
      for group in colors:
        Row, Col, (D0, D1, D2) = group[0]
        if (abs(d0 - D0) + abs(d1 - D1) + abs(d2-D2)) <= 25: # adjust based on color differences, previously 15
          group.append(color)
          found_group = True
          break
      if not found_group:
        colors.append([color])
  colorcoords = []
  for color in colors:
    coord = [[color[0][0], color[0][1]], [color[1][0], color[1][1]]]
    colorcoords.append(coord)
  return colorcoords

def colorsdict(colorcoords):
    colors = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(len(colorcoords)):
        key = letters[i]
        colors[key] = i
    return colors

def board_matrix(colorcoords, rows, cols):
  board = [['.'] * cols for _ in range(rows)]
  letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  for i in range(len(colorcoords)):
      [[x, y], [w, z]] = colorcoords[i]
      key = letters[i]
      board[x][y] = key
      board[w][z] = key
  return board

def board_matrix_string(board_matrix):
  board = []
  for i in range(len(board_matrix)):
    board.append("".join(board_matrix[i]))
  return board

def get_cell_center(row, col, h_lines, v_lines, window_x, window_y, top_crop):
  v1 = v_lines[col]
  v2 = v_lines[col + 1]
  h1 = h_lines[row]
  h2 = h_lines[row + 1]
  x = window_x + (v1 + v2) / 2
  y = window_y + top_crop + (h1 + h2) / 2
  return x, y