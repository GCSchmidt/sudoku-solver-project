import os
import logging
import numpy as np
import pandas as pd
import time
import argparse
from collections import defaultdict
import cv2 as cv

# suppress TensorFlow messsages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import snet

logger = logging.getLogger(__name__)
logger.disabled = True
digit_clf = snet.SNET_Model()

# Helper Functions


def load_quiz_from_dataset(
        quiz_df: pd.DataFrame,
        quiz_numnber: int
        ):
    """
    Load sudoku quiz and its solution from dataset located at ./datasets/sudoku.csv
    (zero-indexed).
    """
    quiz_str = str(quiz_df.loc[quiz_numnber]['quizzes'])
    solution_str = quiz_df.loc[quiz_numnber]['solutions']
    quiz_arr = quiz_str_to_grid(quiz_str)
    return quiz_arr, solution_str


def quiz_str_to_grid(quiz_str: str) -> np.ndarray:
    """
    Converts a sudoku quiz in the string format of the dataset into a np.array
    """
    quiz_arr = np.array([np.uint8(c) for c in quiz_str])
    quiz_arr = quiz_arr.reshape(9, 9)
    return quiz_arr


def read_xlsx_file(quiz_file: str) -> np.ndarray:
    """
    Loads quiz from .xlsx file to np array. 
    """
    df = pd.read_excel(quiz_file,
                       index_col=None,
                       header=None,
                       dtype=None, nrows=9,
                       usecols="A:I")
    grid_initial = df.to_numpy(dtype=np.uint8)
    return grid_initial


def row_and_col_to_region(row: int, col: int) -> int:
    """
    Maps row and col indices to region indices. Zero-based indexing.
    """
    return (row // 3) * 3 + (col // 3)


def sudoku_grid_to_string(grid: np.ndarray) -> str:
    """
    Creates a nice string representation of a sudoku grid.
    """
    formatted_rows = []
    for i, row in enumerate(grid):
        # Format the row with vertical separators for 3x3 blocks
        formatted_row = " | ".join(
            " ".join(str(cell) if cell != 0 else "." for cell in row[j:j+3])
            for j in range(0, 9, 3)
        )
        formatted_rows.append(formatted_row)

        # Add a horizontal separator every 3 rows
        if (i + 1) % 3 == 0 and i != 8:
            formatted_rows.append("-" * 21)  # Length matches the row format

    return "\n".join(formatted_rows)


def filter_close_coordinates(coordinates, min_distance=10):
    """
    Remove coordinates that are too close to each other.
    """
    filtered = []

    for coord_1 in coordinates:
        preserve = True
        for coord_2 in filtered:
            distance = np.linalg.norm(coord_1 - coord_2)
            if distance > min_distance:
                continue
            else:
                preserve = False
                break
        if preserve:
            filtered.append(coord_1) 
    return filtered 


def get_coords_of_lines(corner_coords):
    """
    Determines the coordiantes of the lines within the images from Harris corner coordianates.
    """
    corner_coords.sort(key=lambda coord: coord[0])  # sort based on y coordinate
    vertical_lines_coords = [int(coord[1]) for coord in corner_coords[0:10]]  # get the 10 smallest y coordinates
    vertical_lines_coords.sort()  # the x coordiantes of the vertical lines

    corner_coords.sort(key=lambda coord: coord[1])  # sort based on x coordinate
    horizontal_lines_coords = [int(coord[0]) for coord in corner_coords[0:10]]  # get the 10 smallest x coordinates
    horizontal_lines_coords.sort()  # the y coordiantes of the horizontal lines, y = horizontal_lines
    return vertical_lines_coords, horizontal_lines_coords


def crop_img(img, scale=1.0):
    """
    Crops an image from center according to some scale.
    """
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped


def get_sqr_sub_image(img, row: int, col: int, vert_lines: list, hori_lines: list):
    """
    Get sub image from image, which captures a single cell of a sudoku quiz image.
    """
    x_min = vert_lines[col]
    x_max = vert_lines[col+1]
    y_min = hori_lines[row]
    y_max = hori_lines[row+1]
    sqr_sub_image = img[y_min:y_max, x_min:x_max]
    cropped_image = crop_img(sqr_sub_image, 0.8)
    img_blur = cv.GaussianBlur(cropped_image, (11, 11),0)
    img_th = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 1)
    resized_img = cv.resize(img_th, (32, 32), interpolation=cv.INTER_NEAREST)
    return resized_img


def detect_digit(images) -> str:
    """
    Returns the most likely digit in the image using the trained classifier.
    """
    global digit_clf
    digit_list = digit_clf.predict_digits(images)
    str_digits = "".join([str(digit) for digit in digit_list])
    return str_digits


def image_to_string(img, vertical_lines_coords, horizontal_lines_coords):
    """
    Generates a sudoku quiz string from an image. 
    """
    output_str = ''
    cell_images = []
    for row in range(0, 9):
        for col in range(0, 9):
            cell_img = get_sqr_sub_image(img, row, col, vertical_lines_coords, horizontal_lines_coords)
            cell_images.append(cell_img)
            
    output_str = detect_digit(cell_images)
    return output_str


def image_to_sudoku_quiz(f_path) -> str:
    """
    Generates a sudoku quiz grid from an image. 
    """
    # load image and preprocess
    img_gray = cv.imread(f_path, cv.IMREAD_GRAYSCALE)
    img_th = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 3)

    # detect corner feature
    dst = cv.cornerHarris(img_th, 2, 3, 0.04)
    dst = cv.dilate(dst, None)
    corner_condition = dst > 0.01*dst.max()
    corner_coords = np.argwhere(corner_condition)
    corner_coords_reduced = filter_close_coordinates(corner_coords, 15) # reduce corners

    # get coordinates of lines
    vertical_lines_coords, horizontal_lines_coords = get_coords_of_lines(corner_coords_reduced)

    # generate string 
    raw_string = image_to_string(img_th, vertical_lines_coords, horizontal_lines_coords)

    return quiz_str_to_grid(raw_string)


# Classes


class Solver():

    def __init__(self, grid_initial):
        self.grid_intermediate = grid_initial.copy()
        unsolved_cell_positions = np.argwhere(self.grid_intermediate == 0)
        self.unsolved_cells = defaultdict(lambda: np.arange(1, 10, dtype=np.uint8))  # default value is np array 1-9
        for pos in map(tuple, unsolved_cell_positions):
            self.unsolved_cells[pos]  # create dict entries with defaut values
        self.try_hard = True

    def check_if_only(self, possibilities: np.ndarray, positions: list) -> np.ndarray:
        """
        Checks to see if posibilities contains a solution/number,
        that is not a possibility of the unsolved cells at the given positions.
        """
        for possibility in possibilities:
            found = False
            for pos in positions:
                if possibility in self.unsolved_cells[pos]:
                    logger.info(f"\n\t\tCant tell if {possibility} is correct\n\t\tCell at {pos} could be {self.unsolved_cells[pos]}")
                    found = True
                    break
            if found:
                continue
            else:
                # if not found, return the only option
                logger.info(f"\n\t{possibility} was the only possiblity")
                return np.array([possibility])
        return possibilities

    def check_row(self, grid, position,
                  possibilities=np.arange(1, 10, dtype=np.uint8)):
        """
        Use the row values to reduce possibilites for the cell at the given position.
        """
        row_index = position[0]
        col_index = position[1]
        row_to_check = grid[row_index]
        possibilities = np.array([value for value in possibilities if (value not in row_to_check)])
        # if it is the only possibility in the row
        logger.info(f"\n\tPossibilities after checkig row: {possibilities}")
        # check to see if only possibility in row
        if len(possibilities) > 1 and self.try_hard:
            unsolved_cell_cols = np.argwhere(row_to_check == 0)
            # remove the current cell col
            unsolved_cell_cols = [col[0] for col in unsolved_cell_cols if col != col_index]
            unsolved_cell_positions = [(row_index, col) for col in unsolved_cell_cols]
            logger.info(f"\n\tChecking if only for row")
            possibilities = self.check_if_only(possibilities, unsolved_cell_positions)
        return possibilities

    def check_column(self, grid, position,
                     possibilities=np.arange(1, 10, dtype=np.uint8)):
        """
        Use the column values to reduce possibilites for the cell at the given position.
        """
        row_index = position[0]
        col_index = position[1]
        col_to_check = grid[:, col_index]
        possibilities = np.array([value for value in possibilities if value not in col_to_check])
        logger.info(f"\n\tPossibilities after checking col: {possibilities}")
        # check to see if only possibility in col
        if len(possibilities) > 1 and self.try_hard:
            unsolved_cell_rows = np.argwhere(col_to_check == 0)
            # remove the current cell col
            unsolved_cell_rows = [row[0] for row in unsolved_cell_rows if row != row_index]
            unsolved_cell_positions = [(row, col_index) for row in unsolved_cell_rows]
            logger.info(f"\n\tChecking if only for col")
            possibilities = self.check_if_only(possibilities, unsolved_cell_positions)
        return possibilities

    def check_region(self, grid, position,
                     possibilities=np.arange(1, 10, dtype=np.uint8)):
        """
        Use the region values to reduce possibilites for the cell at the given position.
        """
        row_index = position[0]
        col_index = position[1]
        box_to_check = grid[3*(row_index//3):(3*(row_index//3) + 3), 3*(col_index//3):(3*(col_index//3) + 3)]
        box_to_check_flat = np.unique(box_to_check)
        possibilities = np.array([value for value in possibilities if value not in box_to_check_flat])
        logger.info(f"\n\tPossibilities after checking region: {possibilities}")
        # check to see if only possibility in col
        if len(possibilities) > 1 and self.try_hard:
            unsolved_cell_positions = np.argwhere(box_to_check == 0)
            new_unsolved_cell_positions = []
            base_row = (row_index // 3) * 3
            base_col = (col_index // 3) * 3
            for pos in unsolved_cell_positions:
                new_row, new_col = pos[0] + base_row, pos[1] + base_col
                if (new_row != row_index) or (new_col != col_index):
                    new_unsolved_cell_positions.append((new_row, new_col))
            logger.info(f"\n\tChecking if only for region")
            possibilities = self.check_if_only(possibilities, new_unsolved_cell_positions)
        return possibilities

    def check_complete(self) -> bool:
        """
        Check if all cells are solved:
        """
        return not self.unsolved_cells      # if the dictionary is empty, the solution is complete

    def get_ids_of_unsolved_cells(self):
        """Get the ids of rows, columns and regions with unsolved cells"""
        row_ids, col_ids, region_ids = set(), set(), set() 

        for pos in self.unsolved_cells:
            row_ids.add(pos[0])
            col_ids.add(pos[1])
            row_ids.add((pos[0] // 3) * 3 + (pos[1] // 3))

        return row_ids, col_ids, region_ids

    def get_unsolved_positions_in_row(self, row_id) -> list:
        """Get a list of postions of unsolved cells in a row"""
        positions = [pos for pos in self.unsolved_cells if pos[0] == row_id]
        return positions
    
    def get_unsolved_positions_in_col(self, col_id) -> list:
        """Get a list of postions of unsolved cells in a row"""
        positions = [pos for pos in self.unsolved_cells if pos[1] == col_id]
        return positions

    def get_unsolved_positions_in_region(self, reg_id) -> list:
        """Get a list of postions of unsolved cells in a region
        (top-left=0, bottom-right=8)"""
        positions = [pos for pos in self.unsolved_cells if ((pos[0] // 3) * 3 + (pos[1] // 3) == reg_id)]
        return positions

    def check_naked_pairs(self, positions: list):

        if len(positions) <= 2:
            return

        pair_count = defaultdict(int)

        for pos in positions:
            possibilities = self.unsolved_cells[pos]
            if len(possibilities) == 2:
                pair_count[tuple(possibilities)] += 1
            
        # only pairs are valid
        pair_count = {key: val for key, val in pair_count.items() if val==2}

        if not pair_count:
            # no pairs found
            return
        
        for pos in positions:
            possibilities = self.unsolved_cells[pos]
            logger.info(f"\nTrying to solve cell @ {pos}\n\tCurrent possibilities: {possibilities}")
            if tuple(self.unsolved_cells[pos]) in pair_count:
                continue
            else:
                for pair in pair_count:
                    # reduce the possibilties to exclude the naked pairs
                    possibilities = self.unsolved_cells[pos]
                    possibilities = np.array([p for p in possibilities if p not in pair]).flatten()
                    self.unsolved_cells[pos] = possibilities
                    logger.info(f"\n\tPossibilities after checkig naked pairs: {self.unsolved_cells[pos]}")
                    if len(possibilities) == 1:
                        # solution found by reducing with naked pairs
                        self.solve_cell(pos, possibilities)
                        self.unsolved_cells.pop(pos, None)
                        break
        return

    def solve_cell(self, position, possibilities, solved_cells=None):
        self.grid_intermediate[position[0]][position[1]] = possibilities[0] # save solution to grid
        if solved_cells is not None:
            solved_cells.append(position) # add key/position to list that we use to remove unsolved cell entry 
        logger.info(f"\nSOLVED!!!")
        return

    def is_sudoku_solved(self):
        """
        Checks if solution is correct
        Time complextity: 9*9 = C
        Space complexity: 3*9*9 = C
        """
        # sets to keep track of what values have been seen
        row_sets = [set() for _ in range(9)]
        col_sets = [set() for _ in range(9)]
        region_sets = [set() for _ in range(9)]

        # start looking at values
        for row in range(0, 9):
            for col in range(0, 9):
                cell_value = self.grid_intermediate[row, col]
                # if invalid number
                if cell_value < 1 or cell_value > 9:
                    logger.info(f"INVALID NUMBER FOUND @{row,col}")
                    return False
                # if value already in row/col/region
                if ((cell_value in row_sets[row]) 
                        or (cell_value in col_sets[col]) 
                        or (cell_value in region_sets[row_and_col_to_region(row, col)])):
                    logger.info(f"DUPLICATE FOUND @{row,col}")
                    return False
                
                # add seen value to corresponding set
                row_sets[row].add(cell_value)
                col_sets[col].add(cell_value)
                region_sets[row_and_col_to_region(row, col)].add(cell_value)
        
        logger.info(
            f"\nSets used to verify validity of solution:"
            f"\nROWS: {row_sets}\n{"-"*100}"
            f"\nCOLS:{col_sets}\n{"-"*100}"
            f"\nREGIONS:{region_sets}")
        return True

    def is_sudoku_solved_2(self):
        """
        Checks if solution is correct
        Time complextity: 9*9 = C
        Space complexity: 3*9*9 = C
        """
        # Check rows
        for row in self.grid_intermediate:
            if not np.array_equal(np.sort(row), np.arange(1, 10)):
                return False

        # Check columns
        for col in self.grid_intermediate.T:  # Transpose to iterate over columns
            if not np.array_equal(np.sort(col), np.arange(1, 10)):
                return False

        # Check regions
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                region = self.grid_intermediate[i:i+3, j:j+3].flatten()
                if not np.array_equal(np.sort(region), np.arange(1, 10)):
                    return False

        return True

    def solve(self) -> np.ndarray:
        """
        Solves the initial grid.
        """
        # TODO check other unsolved cells in row, and collumns and boxes to figure out correct value
        
        loops_with_no_update = 0
        n_unsolved_cells = len(self.unsolved_cells)

        while not self.check_complete():
            solved_cells = []
            for position, possibilities in self.unsolved_cells.items():
                logger.info(f"\nTrying to solve cell @ {position}\n\tCurrent possibilities: {possibilities}")
                # use row values to reduce posibilities
                updated_possibilities = possibilities
                updated_possibilities = self.check_row(grid=self.grid_intermediate,
                                                       position=position,
                                                       possibilities=updated_possibilities)
                # check if already solved
                if len(updated_possibilities) == 1:
                    self.solve_cell(position, updated_possibilities, solved_cells)
                    continue

                updated_possibilities = self.check_column(grid=self.grid_intermediate,
                                                          position=position,
                                                          possibilities=updated_possibilities)
                # check if already solved
                if len(updated_possibilities) == 1:
                    self.solve_cell(position, updated_possibilities, solved_cells)
                    continue

                updated_possibilities = self.check_region(grid=self.grid_intermediate,
                                                          position=position,
                                                          possibilities=updated_possibilities)
                # check if already solved
                if len(updated_possibilities) == 1:
                    self.solve_cell(position, updated_possibilities, solved_cells)
                    continue

                logger.info(f"\n\tCurrent possibilities after all checks: {updated_possibilities}")
                self.unsolved_cells[position] = updated_possibilities

            for position in solved_cells:
                self.unsolved_cells.pop(position, None)
            
            if len(self.unsolved_cells) < n_unsolved_cells:
                n_unsolved_cells = len(self.unsolved_cells)
                loops_with_no_update = 0
            else:
                loops_with_no_update += 1
                # no changes were made, try with naked pairs
                row_ids, col_ids, region_ids = self.get_ids_of_unsolved_cells()
                for i in row_ids:
                    self.check_naked_pairs(self.get_unsolved_positions_in_row(i))
                for i in col_ids:
                    self.check_naked_pairs(self.get_unsolved_positions_in_col(i))
                for i in region_ids:
                    self.check_naked_pairs(self.get_unsolved_positions_in_region(i))

            logger.info(f"\nNUMBER OF LOOPS WITH NO UPDATE: {loops_with_no_update}")
            
            if loops_with_no_update >= 2:
                logger.info(f"\nTOO MANY LOOPS")
                logger.info(f"\nATTEMPTING TO SOLVE WITH BRUTE FORCE")
                # use brute force to find solution
                # needs to be optimized
                self.solve_recursively(0, 0)
                break

        return self.grid_intermediate
    
    ############################################
    # functions for brute force solution 
    # copied from https://www.geeksforgeeks.org/dsa/sudoku-backtracking-7/
    # TODO increase efficiency
    ############################################

    def is_valid_cell_value(self, row, col, num):
        """Check to see if num can be placed at the cell at (row,col)."""
        # Check if num exists in the row
        for x in range(9):
            if self.grid_intermediate[row][x] == num:
                return False

        # Check if num exists in the col
        for x in range(9):
            if self.grid_intermediate[x][col] == num:
                return False

        # Check if num exists in the 3x3 sub-matrix
        startRow = row - (row % 3)
        startCol = col - (col % 3)

        for i in range(3):
            for j in range(3):
                if self.grid_intermediate[i + startRow][j + startCol] == num:
                    return False

        return True
    
    def solve_recursively(self, row, col):
        # base case: Reached nth column of the last row
        if row == 8 and col == 9:
            return True

        # If last column of the row go to the next row
        if col == 9:
            row += 1
            col = 0

        # If cell is already occupied then move forward
        if self.grid_intermediate[row][col] != 0:
            return self.solve_recursively(row, col + 1)

        for num in range(1, 10):
            # If it is safe to place num at current position
            if self.is_valid_cell_value(row, col, num):
                self.grid_intermediate[row][col] = num
                if self.solve_recursively(row, col + 1):
                    return True
                self.grid_intermediate[row][col] = 0

        return False


def main(args):
    """
    Main
    """
    quiz_file, log = args.quiz_file, args.log

    if log:
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(filename=r'./logs/ss.log',
                            filemode='w',
                            level=logging.INFO)
        logger.disabled = False

    try:
        _, file_extension = os.path.splitext(quiz_file)
        if file_extension in ['.png', '.jpeg']:
            grid_initial = image_to_sudoku_quiz(quiz_file)
        elif file_extension == '.xlsx':
            grid_initial = read_xlsx_file(quiz_file)
        else:
            print(f"\'{quiz_file}\' is not in a supported file format.\
                  \nSupported file formats are: png, jpeg and xlsx")
            quit()
    except OSError:
        print(f"Could not open/read file: \'{quiz_file}\'")
        quit()

    print(f"Initial Grid:\n{sudoku_grid_to_string(grid_initial)}")

    solver = Solver(grid_initial)
    t1 = time.time()
    solution_gid = solver.solve()
    t1 = time.time()-t1
    t2 = time.time() 
    solved = solver.is_sudoku_solved()
    t2 = time.time()-t2
    print(f"Solution Grid:\n{sudoku_grid_to_string(solution_gid)}")
    print(f"The solution is {'correct' if solved else 'incorrect'}!")
    print(f"Time to find the solution: {round(t1, 3)}")
    print(f"Time to check the solution: {round(t2, 6)}")

    if not solved:
        print(f"Failed to solve {len(solver.unsolved_cells)} cells."
              "\nTo see possibilities of the unsolved cells view: \'outputs/unsolved_cells.txt\'")
        str_out = "Unsolved cells and possibilities:"
        for position, possibilities in solver.unsolved_cells.items():
            str_out += f"\n\t cell @ [{position[0]},{position[1]}]: {possibilities}"

        os.makedirs('outputs', exist_ok=True)
        with open(r'outputs/unsolved_cells.txt', 'w') as f:
            f.write(str_out)


def parse_arguments():
    """
    Pasre arguments
    """
    parser = argparse.ArgumentParser(
                    prog='SUDOKU SOLVER',
                    description='This program solves SUDOKU quizzes',
                    epilog='Have fun!')

    parser.add_argument('quiz_file', help="image or .xlsx file that has a SUDOKU quiz")  # positional argument
    parser.add_argument('--log', action='store_true', help="Enables logging to file")    # on/off flag
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
