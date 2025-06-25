import os
import sys
import operator
import itertools
from datetime import datetime
from argparse import ArgumentParser
from collections import defaultdict
import pycosat
from functools import reduce

LEFT, RIGHT, TOP, BOTTOM = 1, 2, 4, 8
LR = LEFT | RIGHT
TB = TOP | BOTTOM
TL = TOP | LEFT
TR = TOP | RIGHT
BL = BOTTOM | LEFT
BR = BOTTOM | RIGHT
DELTAS = [(LEFT, 0, -1), (RIGHT, 0, 1), (TOP, -1, 0), (BOTTOM, 1, 0)]
DIRECTION_TYPES = [LR, TB, TL, TR, BL, BR]
DIRECTION_FLIP = {LEFT: RIGHT, RIGHT: LEFT, TOP: BOTTOM, BOTTOM: TOP}
RESULT_STRINGS = dict(s='successful', f='failed', u='unsolvable')

def pair_combinations(collection):
    return itertools.combinations(collection, 2)

def negative_clauses(sat_variables):
    return ((-a, -b) for (a, b) in pair_combinations(sat_variables))

def iterate(puzzle):
    for i, row in enumerate(puzzle):
        for j, char in enumerate(row):
            yield i, j, char

def valid_position(rows, cols, i, j):
    return 0 <= i < rows and 0 <= j < cols

def neighbor_cells(i, j):
    return ((direction, i + delta_x, j + delta_y) for (direction, delta_x, delta_y) in DELTAS)
    
def valid_neighbors(rows, cols, i, j):
    return ((direction, ni, nj) for (direction, ni, nj) in neighbor_cells(i, j) if valid_position(rows, cols, ni, nj))
    
def make_color_clauses(puzzle, colors, generate_color_id):
    color_clauses = []
    num_colors = len(colors)
    rows = len(list(puzzle))
    cols = len(list(puzzle[0]))
    for i, j, char in iterate(puzzle):
        if char.isalnum():
            endpoint_id = colors[char]
            color_clauses.append([generate_color_id(i, j, endpoint_id)])
            for new_color in range(num_colors):
                if new_color != endpoint_id:
                    color_clauses.append([-generate_color_id(i, j, new_color)])
            neighbor_vars = [generate_color_id(ni, nj, endpoint_id) for _, ni, nj in valid_neighbors(rows, cols, i, j)]
            color_clauses.append(neighbor_vars)
            color_clauses.extend(negative_clauses(neighbor_vars))
        else:
            color_clauses.append([generate_color_id(i, j, color) for color in range(num_colors)])
            cell_color_ids = (generate_color_id(i, j, color) for color in range(num_colors))
            color_clauses.extend(negative_clauses(cell_color_ids))
    return color_clauses

def make_directional_variables(puzzle, start_variable, colors):
    rows = len(list(puzzle))
    cols = len(list(puzzle[0]))
    direction_variables = dict()
    num_dir_vars = 0
    for i, j, char in iterate(puzzle):
        if char.isalnum():
            continue
        neighbor_bits = (direction for (direction, ni, nj) in valid_neighbors(rows, cols, i, j))
        bit_reduction = reduce(operator.or_, neighbor_bits, 0)
        direction_variables[i, j] = dict()
        for direction_type in DIRECTION_TYPES:
            if bit_reduction & direction_type == direction_type:
                num_dir_vars += 1
                direction_variables[i, j][direction_type] = start_variable + num_dir_vars
    return direction_variables, num_dir_vars

def make_directional_clauses(puzzle, colors, generate_color_id, direction_variables):
    direction_clauses = []
    num_colors = len(colors)
    rows = len(list(puzzle))
    cols = len(list(puzzle[0]))
    for i, j, char in iterate(puzzle):
        if char.isalnum():
            continue
        cell_directions = direction_variables[(i, j)]
        cell_direction_ids = cell_directions.values()
        direction_clauses.append(cell_direction_ids)
        direction_clauses.extend(negative_clauses(cell_direction_ids))
        for color in range(num_colors):
            color_id_1 = generate_color_id(i, j, color)
            for direction_bit, ni, nj in neighbor_cells(i, j):
                color_id_2 = generate_color_id(ni, nj, color)
                for direction_type, direction_id in cell_directions.items():
                    if direction_type & direction_bit:
                        direction_clauses.append([-direction_id, -color_id_1, color_id_2])
                        direction_clauses.append([-direction_id, color_id_1, -color_id_2])
                    elif valid_position(rows, cols, ni, nj):
                        direction_clauses.append([-direction_id, -color_id_1, -color_id_2])
    return direction_clauses

def sat_reduction(puzzle, colors):
    rows = len(list(puzzle))
    cols = len(list(puzzle[0]))
    num_colors = len(colors)
    num_cells = rows * cols
    total_ids = num_colors * num_cells
    def generate_color_id(i, j, color):
        return (i * cols + j) * num_colors + color + 1
    start = datetime.now()
    color_clauses = make_color_clauses(puzzle, colors, generate_color_id)
    direction_variables, num_direction_variables = make_directional_variables(puzzle, total_ids, colors)
    dir_clauses = make_directional_clauses(puzzle, colors, generate_color_id, direction_variables)
    total_vars = total_ids + num_direction_variables
    total_clauses = color_clauses + dir_clauses
    reduction_time = (datetime.now() - start).total_seconds()
    return generate_color_id, direction_variables, total_vars, total_clauses, reduction_time

def decode_solution(puzzle, colors, generate_color_id, direction_variables, solution):
    solution = set(solution)
    num_colors = len(colors)
    decoded_board = []
    for i, row in enumerate(puzzle):
        decoded_row = []
        for j, char in enumerate(row):
            cell_color = -1
            for color in range(num_colors):
                if generate_color_id(i, j, color) in solution:
                    cell_color = color
            cell_direction = -1
            if not char.isalnum():
                for direction_type, direction_variable in direction_variables[i, j].items():
                    if direction_variable in solution:
                        cell_direction = direction_type
            decoded_row.append((cell_color, cell_direction))
        decoded_board.append(decoded_row)
    return decoded_board

def trace_path(decoded, visited, i, j):
    rows = len(decoded)
    cols = len(decoded[0])
    path = []
    is_cycle = False
    prev_i, prev_j = -1, -1
    while True:
        advanced = False
        color, direction_type = decoded[i][j]
        visited[i][j] = 1
        path.append((i, j))
        for direction_bit, ni, nj in valid_neighbors(rows, cols, i, j):
            if (ni, nj) == (prev_i, prev_j):
                continue
            n_color, n_direction_type = decoded[ni][nj]
            if ((direction_type >= 0 and (direction_type & direction_bit)) or
                    (direction_type == -1 and n_direction_type >= 0 and
                     n_direction_type & DIRECTION_FLIP[direction_bit])): 
                if visited[ni][nj]:
                    is_cycle = True
                else:
                    prev_i, prev_j = i, j
                    i, j = ni, nj
                    advanced = True
                break
        if not advanced:
            break
    return path, is_cycle

def extract_paths(decoded, direction_variables):
    rows = len(decoded)
    cols = len(decoded[0])
    seen_colors = set()
    visited = [[0] * cols for _ in range(rows)]
    all_paths = []
    for i, j, (color, direction_type) in iterate(decoded):
        if direction_type == -1 and color not in seen_colors:
            seen_colors.add(color)
            path, is_cycle = trace_path(decoded, visited, i, j)
            all_paths.append(path)
    return all_paths


def detect_cycles(decoded, direction_variables):
    rows = len(decoded)
    cols = len(decoded[0])
    seen_colors = set()
    visited = [[0] * cols for _ in range(rows)]
    for i, j, (color, direction_type) in iterate(decoded):
        if direction_type == -1 and color not in seen_colors:
            seen_colors.add(color)
            path, is_cycle = trace_path(decoded, visited, i, j)
    extra_clauses = []
    for i, j in itertools.product(range(rows), range(cols)):
        if not visited[i][j]:
            path, is_cycle = trace_path(decoded, visited, i, j)
            clause = []
            for ri, rj in path:
                _, direction_type = decoded[ri][rj]
                direction_variable = direction_variables[ri, rj][direction_type]
                clause.append(-direction_variable)
            extra_clauses.append(clause)
    return extra_clauses

def show_solution(colors, decoded):
  color_chars = [None] * len(colors)
  for char, color in colors.items():
      color_chars[color] = char
  output_lines = []
  for decoded_row in decoded:
      line = ''
      for (color, direction_type) in decoded_row:
          color_char = color_chars[color]
          line += color_char
      output_lines.append(line)
  return '\n'.join(output_lines)

def solve_sat(puzzle, colors, generate_color_id, direction_variables, clauses):
    start = datetime.now()
    decoded_solution = None
    all_decoded = []
    repairs = 0
    while True:
        solution = pycosat.solve(clauses)
        if not isinstance(solution, list):
            decoded_solution = None
            all_decoded.append(decoded_solution)
            break
        decoded_solution = decode_solution(puzzle, colors, generate_color_id, direction_variables, solution)
        all_decoded.append(decoded_solution)
        extra_clauses = detect_cycles(decoded_solution, direction_variables)
        if not extra_clauses:
            break
        clauses += extra_clauses
        repairs += 1
    solve_time = (datetime.now() - start).total_seconds()
    if decoded_solution is None:
      print(f'Solver returned {str(solution)} after {repairs:,} cycle repair(s) and {solve_time:.3f} seconds')
    else:
      solved_board = show_solution(colors, decoded_solution)
    paths = extract_paths(decoded_solution, direction_variables)
    return solution, decoded_solution, repairs, solve_time, solved_board, paths

def pyflow_solver_main(puzzle, colors):
    generate_color_id, direction_variables, total_variables, total_clauses, reduction_time = sat_reduction(puzzle, colors)
    solution, _, repairs, solve_time, solved_board, paths = solve_sat(puzzle, colors, generate_color_id, direction_variables, total_clauses)
    total_time = reduction_time + solve_time
    return solved_board, paths