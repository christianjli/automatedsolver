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

DELTAS = [(LEFT, 0, -1), (RIGHT, 0, 1), (TOP, -1, 0), (BOTTOM, 1, 0)]

LR = LEFT | RIGHT
TB = TOP | BOTTOM
TL = TOP | LEFT
TR = TOP | RIGHT
BL = BOTTOM | LEFT
BR = BOTTOM | RIGHT

DIR_TYPES = [LR, TB, TL, TR, BL, BR]

DIR_FLIP = {LEFT: RIGHT, RIGHT: LEFT, TOP: BOTTOM, BOTTOM: TOP}

RESULT_STRINGS = dict(s='successful', f='failed', u='unsolvable')

def all_pairs(collection):
    ''' Returns all combinations of pairs from a collection '''
    return itertools.combinations(collection, 2)

def no_two(satvars):
    ''' Generates clauses from collection of SAT variables
    so that no two can both be true '''
    return ((-a, -b) for (a, b) in all_pairs(satvars))

def explode(puzzle):
    ''' Clean iterator for coordinate, value '''
    for i, row in enumerate(puzzle):
        for j, char in enumerate(row):
            yield i, j, char

def valid_pos(rows, cols, i, j):
    ''' Check if coordinate within bounds '''
    return 0 <= i < rows and 0 <= j < cols

def all_neighbors(i, j):
    ''' Returns all neighbors at given coordinate '''
    return ((direction, i + delta_x, j + delta_y)
            for (direction, delta_x, delta_y) in DELTAS)
    
def valid_neighbors(rows, cols, i, j):
    ''' Returns all valid neighbors '''
    return ((direction, ni, nj) for (direction, ni, nj)
            in all_neighbors(i, j) if valid_pos(rows, cols, ni, nj))
    
def make_color_clauses(puzzle, colors, color_var):
    ''' Generates CNF clauses with (# cells) * (# colors) SAT variables, 
    each cell has one color, one hot '''
    clauses = [] # lay foundation of the problem for SAT solver
    num_colors = len(colors)
    rows = len(list(puzzle))
    cols = len(list(puzzle[0]))

    for i, j, char in explode(puzzle): # each cell
        if char.isalnum(): # flow endpoint
            endpoint_id = colors[char] # index value
            # add unique id through color_var at predetermined endpoint
            clauses.append([color_var(i, j, endpoint_id)])

            for other_color in range(num_colors):
                if other_color != endpoint_id: # all other colors
                    # add negative clauses, only one color per coord
                    clauses.append([-color_var(i, j, other_color)])

            # path must continue to valid neighbor
            neighbor_vars = [color_var(ni, nj, endpoint_id) for
                             _, ni, nj in valid_neighbors(rows, cols, i, j)]

            # guaranteed one neighbor will have this color
            clauses.append(neighbor_vars)
            # make sure only one neighbor has this color through no_two
            clauses.extend(no_two(neighbor_vars))

        else:
            # currently empty cell, add clauses for all colors
            clauses.append([color_var(i, j, color)
                            for color in range(num_colors)])

            # generate vars for all colors
            cell_color_vars = (color_var(i, j, color) for
                               color in range(num_colors))

            # make sure only one color fills empty cell through no_two
            clauses.extend(no_two(cell_color_vars))

    return clauses

def make_dir_vars(puzzle, start_var, colors):
    ''' Generates direction SAT variables '''
    rows = len(list(puzzle))
    cols = len(list(puzzle[0]))
    dir_vars = dict()
    num_dir_vars = 0

    for i, j, char in explode(puzzle):
        if char.isalnum(): # predetermined endpoint
            continue

        # get all valid directional bits for given cell
        neighbor_bits = (direction for (direction, ni, nj)
                         in valid_neighbors(rows, cols, i, j))

        # use OR operator to represent all possible directional values
        cell_flags = reduce(operator.or_, neighbor_bits, 0)
        # new dictionary of coordinates within dictionary dir_vars
        # stores directional SAT variable identifiers
        dir_vars[i, j] = dict()

        for code in DIR_TYPES:
            # if current direction is represented in cell_flags identifier
            # value (Left/Right, Top/Bottom, etc) max of 6
            if cell_flags & code == code:
                num_dir_vars += 1
                # start variable offsets from last unique id in color_var
                dir_vars[i, j][code] = start_var + num_dir_vars

    return dir_vars, num_dir_vars

def make_dir_clauses(puzzle, colors, color_var, dir_vars):
    ''' Generate clauses for color and direction SAT variables, 
    exactly one direction implying matching colors '''

    dir_clauses = []
    num_colors = len(colors)
    rows = len(list(puzzle))
    cols = len(list(puzzle[0]))

    for i, j, char in explode(puzzle):
        if char.isalnum(): # predetermined endpoint
            continue

        # dictionary of dir: id from dir_vars in previous function
        cell_dir_dict = dir_vars[(i, j)]
        # all id values from cell_dir_dict
        cell_dir_vars = cell_dir_dict.values()

        # adds possible directions to clauses
        dir_clauses.append(cell_dir_vars)

        # ensures only one direction chosen through no_two
        dir_clauses.extend(no_two(cell_dir_vars))

        for color in range(num_colors):
            # get unique SAT identifier for this cell
            color_1 = color_var(i, j, color)

            for dir_bit, n_i, n_j in all_neighbors(i, j):
                # get color var for each neighboring cell
                color_2 = color_var(n_i, n_j, color)

                # for each direction variable and identifier in this cell
                for dir_type, dir_var in cell_dir_dict.items():

                    # if this cell points at current neighbor
                    if dir_type & dir_bit:
                        # because we already ensured no_two
                        # if directions match, colors match
                        dir_clauses.append([-dir_var, -color_1, color_2])
                        dir_clauses.append([-dir_var, color_1, -color_2])
                    elif valid_pos(rows, cols, n_i, n_j):
                        # otherwise colors don't match
                        dir_clauses.append([-dir_var, -color_1, -color_2])

    return dir_clauses

def reduce_to_sat(puzzle, colors):
    ''' Turns puzzle to SAT problem, returns clauses '''

    rows = len(list(puzzle))
    cols = len(list(puzzle[0]))
    num_colors = len(colors)
    num_cells = rows * cols
    num_color_vars = num_colors * num_cells

    def color_var(i, j, color):
        ''' Generates unique SAT identifier for cell coordinate and color '''
        return (i * cols + j) * num_colors + color + 1

    start = datetime.now()

    # get all color clauses for SAT
    color_clauses = make_color_clauses(puzzle, colors, color_var)
    # dir_vars is dictionary of coord : dir : id
    # num_dir_vars is # directional variables
    dir_vars, num_dir_vars = make_dir_vars(puzzle, num_color_vars, colors)
    # all directional clauses for SAT
    dir_clauses = make_dir_clauses(puzzle, colors, color_var, dir_vars)

    num_vars = num_color_vars + num_dir_vars # total variables
    clauses = color_clauses + dir_clauses # total clauses

    reduce_time = (datetime.now() - start).total_seconds()
    # print(f'generated {len(color_clauses):,} clauses over {num_color_vars:,} color variables')
    # print(f'generated {len(dir_clauses):,} dir clauses over {num_dir_vars:,} dir variables')
    # print(f'total {len(clauses):,} clauses over {num_vars:,} variables')
    # print(f'reduced to SAT in {reduce_time:.3f} seconds')

    return color_var, dir_vars, num_vars, clauses, reduce_time

def decode_solution(puzzle, colors, color_var, dir_vars, sol):
    ''' Decodes SAT solution by taking advantage of one-hot encoding,
    returning (color, direction) pairs '''

    sol = set(sol) # sol is SAT solution
    num_colors = len(colors)

    decoded = []

    for i, row in enumerate(puzzle):

        decoded_row = []

        for j, char in enumerate(row):

            # use one-hot encoding to determine color for this cell
            cell_color = -1

            for color in range(num_colors):
                # only true for one color, which we assign to cell_color
                if color_var(i, j, color) in sol:
                    assert cell_color == -1
                    cell_color = color # match 

            assert cell_color != -1 # ensure cell_color was matched

            cell_dir_type = -1 # repeat with direction

            if not char.isalnum(): # not predetermined endpoint
                # only true for one direction, assign it to cell_dir_type
                for dir_type, dir_var in dir_vars[i, j].items():
                    if dir_var in sol:
                        assert cell_dir_type == -1
                        cell_dir_type = dir_type

                assert cell_dir_type != -1

            # cell_dir_type stays -1 if endpoint
            decoded_row.append((cell_color, cell_dir_type))

        decoded.append(decoded_row)

    return decoded

def make_path(decoded, visited, cur_i, cur_j):
    ''' Follow path from given coordinates, until we reach 
    an endpoint, make no advancement, or find a cycle '''

    rows = len(decoded)
    cols = len(decoded[0])

    run = [] # stores the path of each flow
    is_cycle = False
    prev_i, prev_j = -1, -1

    while True:

        advanced = False

        # get color and direction for current cell
        color, dir_type = decoded[cur_i][cur_j]
        # mark as visited
        visited[cur_i][cur_j] = 1
        # add to run
        run.append((cur_i, cur_j))

        # check each valid neighbor
        for dir_bit, n_i, n_j in valid_neighbors(rows, cols, cur_i, cur_j):

            # don't go backwards
            if (n_i, n_j) == (prev_i, prev_j):
                continue

            # get neighbor color and direction
            n_color, n_dir_type = decoded[n_i][n_j]

            # if current cell points at valid neighbor, or
            # neighbor points at current (endpoint) cell
            if ((dir_type >= 0 and (dir_type & dir_bit)) or
                    (dir_type == -1 and n_dir_type >= 0 and
                     n_dir_type & DIR_FLIP[dir_bit])): 
                     

                # must be same color
                assert color == n_color

                # detected cycle
                if visited[n_i][n_j]:
                    is_cycle = True
                else:
                    # update variables, continue path
                    prev_i, prev_j = cur_i, cur_j
                    cur_i, cur_j = n_i, n_j
                    advanced = True

                # cycle detected or path advanced, don't need to
                # check other neighbors
                break

        # went through all neighbors, didn't advance or found cycle, quit
        if not advanced:
            break

    return run, is_cycle

def extract_paths(decoded, dir_vars):
    rows = len(decoded)
    cols = len(decoded[0])
    colors_seen = set()
    visited = [[0] * cols for _ in range(rows)]
    all_paths = []

    for i, j, (color, dir_type) in explode(decoded):
        # If this is an endpoint for a new color
        if dir_type == -1 and color not in colors_seen:
            colors_seen.add(color)
            run, is_cycle = make_path(decoded, visited, i, j)
            assert not is_cycle
            all_paths.append(run)

    return all_paths


def detect_cycles(decoded, dir_vars):

    ''' Checks SAT solution for existing cycles, returns new 
    clauses to prevent cycles on next repair '''

    rows = len(decoded)
    cols = len(decoded[0])
    colors_seen = set()
    visited = [[0] * cols for _ in range(rows)]

    for i, j, (color, dir_type) in explode(decoded):

        # flow endpoint for a color not yet seen
        if dir_type == -1 and color not in colors_seen:

            # now seen
            assert not visited[i][j]
            colors_seen.add(color)

            # now visited
            run, is_cycle = make_path(decoded, visited, i, j)
            assert not is_cycle

    # check for unvisited cells, means there's a cycle
    extra_clauses = []

    for i, j in itertools.product(range(rows), range(cols)):

        if not visited[i][j]:

            # get run, (x, y) coords for the path, and if it's a cycle
            run, is_cycle = make_path(decoded, visited, i, j)
            assert is_cycle

            # generate a negative clause for the path
            clause = []

            for r_i, r_j in run:
                # decoded has color, dir_type
                _, dir_type = decoded[r_i][r_j]
                # get unique SAT variable
                dir_var = dir_vars[r_i, r_j][dir_type]
                # add negative clause to prevent same cycle repeating
                clause.append(-dir_var)

            extra_clauses.append(clause)

    return extra_clauses

def show_solution(colors, decoded):
  ''' returns solution of the board '''
  color_chars = [None] * len(colors)
  for char, color in colors.items():
      color_chars[color] = char

  output_lines = []

  for decoded_row in decoded:
      line = ''
      for (color, dir_type) in decoded_row:
          color_char = color_chars[color]
          line += color_char
      output_lines.append(line)
  return '\n'.join(output_lines)

def solve_sat(puzzle, colors, color_var, dir_vars, clauses):
    ''' Solve SAT with given clauses, if cycles are found, we repair 
    and repeat, returning the solution, decoded puzzle solution, 
    and cycle repairs needed '''

    start = datetime.now()

    decoded = None
    all_decoded = []
    repairs = 0

    while True:

        sol = pycosat.solve(clauses) # SAT solver

        # check if it returns a list, otherwise no valid solution
        if not isinstance(sol, list):
            decoded = None
            all_decoded.append(decoded)
            break

        # turn SAT solution with IDs back to understandable pathways
        decoded = decode_solution(puzzle, colors, color_var, dir_vars, sol)
        all_decoded.append(decoded)

        # if cycles detected, add extra clauses
        extra_clauses = detect_cycles(decoded, dir_vars)

        if not extra_clauses:
            break # done if no cycles

        clauses += extra_clauses
        repairs += 1 # retry with new clauses

    solve_time = (datetime.now() - start).total_seconds()

    # to display cycles
    # print(f'Intermediate solution with cycles: {show_solution(options, colors, cycle_decoded)}')
    if decoded is None:
      print(f'Solver returned {str(sol)} after {repairs:,} cycle repair(s) and {solve_time:.3f} seconds')
    else:
      # print(f'Obtained solution after {repairs:,} cycle repair(s) and {solve_time:3f} seconds')
      solved_board = show_solution(colors, decoded)

    paths = extract_paths(decoded, dir_vars)
    return sol, decoded, repairs, solve_time, solved_board, paths

def pyflow_solver_main(puzzle, colors):
    ''' Main solver '''

    stats = dict()

    color_var, dir_vars, num_vars, clauses, reduce_time = \
        reduce_to_sat(puzzle, colors) # reduce to SAT outputs

    sol, _, repairs, solve_time, solved_board, paths = solve_sat(puzzle, colors,
                                            color_var, dir_vars, clauses) # solved SAT outputs

    total_time = reduce_time + solve_time

    if isinstance(sol, list):
        result_char = 's'
    elif str(sol) == 'UNSAT':
        result_char = 'u'
    else:
        result_char = 'f'

    cur_stats = dict(repairs=repairs,
                      reduce_time=reduce_time,
                      solve_time=solve_time,
                      total_time=total_time,
                      num_vars=num_vars,
                      num_clauses=len(clauses),
                      count=1)

    # if result not seen before, add to stats dictionary
    if not result_char in stats:
        stats[result_char] = cur_stats

    else:
        # add new stats to dictionary under appropriate result
        for key in cur_stats.keys():
            stats[result_char][key] += cur_stats[key]

    # print(f'finished in total of {total_time:.3f} seconds')
    return solved_board, paths