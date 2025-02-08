import os
import logging
import numpy as np
import pandas as pd
import time
import argparse
from collections import defaultdict


logger = logging.getLogger(__name__)
logger.disabled = True

# Helper Functions


def load_quiz_from_dataset(
        quiz_df: pd.DataFrame,
        quiz_numnber: int
        ):
    """
    Load sudoku quiz and its solution from dataset located at ./datasets/sudoku.csv
    (zero-indexed).
    """
    quiz_str = quiz_df.loc[quiz_numnber]['quizzes']
    solution_str = quiz_df.loc[quiz_numnber]['solutions']
    quiz_arr = quiz_str_to_grid(quiz_str)
    return quiz_arr, solution_str


def quiz_str_to_grid(quiz_str: str) -> np.array:
    """
    Converts a sudoku quiz in the string format of the dataset into a np.array
    """
    quiz_arr = np.array([np.uint8(c) for c in quiz_str])
    quiz_arr = quiz_arr.reshape(9, 9)
    return quiz_arr


def read_input_file(quiz_file: str) -> np.array:
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


def sudoku_grid_to_string(grid: np.array) -> str:
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


# Classes


class Solver():

    def __init__(self, grid_initial):
        self.grid_intermediate = grid_initial.copy()
        unsolved_cell_positions = np.argwhere(self.grid_intermediate == 0)
        self.unsolved_cells = defaultdict(lambda: np.arange(1, 10, dtype=np.uint8))  # default value is np array 1-9
        for pos in map(tuple, unsolved_cell_positions):
            self.unsolved_cells[pos]  # create dict entries with defaut values
        self.try_hard = True

    def check_if_only(self, possibilities: np.array, positions: list) -> np.array:
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

    def solve_cell(self, position, possibilities, solved_cells):
        self.grid_intermediate[position[0]][position[1]] = possibilities[0] # save solution to grid
        solved_cells.append(position) # add key/position to list that we use to remove unsolved cell entry 
        logger.info(f"\nSOLVED!!!")
        return

    def is_sudoku_solved_1(self):
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
                if cell_value in row_sets[row] or cell_value in col_sets[col] or cell_value in region_sets[row_and_col_to_region(row, col)]:
                    logger.info(f"DUPLICATE FOUND @{row,col}")
                    return False
                
                # add seen value to corresponding set
                row_sets[row].add(cell_value)
                col_sets[col].add(cell_value)
                region_sets[row_and_col_to_region(row, col)].add(cell_value)
                logger.info(f'{row_sets}\n{"-"*100}\n{col_sets}\n{"-"*100}\n{region_sets}')
        
        logger.info(f'{row_sets}\n{"-"*100}\n{col_sets}\n{"-"*100}\n{region_sets}')
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

    def solve(self) -> np.array:
        """
        Solves the initial grid.
        """
        # TODO check other unsolved cells in row, and collumns and boxes to figure out correct value
        repeated = 0
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

            repeated += 1
            logger.info(f"\nREPEATED LOOPS: {repeated}")
            if repeated >= 20:
                logger.info(f"\nTOO MANY LOOPS")
                break

        return self.grid_intermediate


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
        grid_initial = read_input_file(quiz_file)
    except OSError:
        print(f"Could not open/read file: \'{quiz_file}\'")
        quit()

    print(f"Initial Grid:\n{sudoku_grid_to_string(grid_initial)}")

    solver = Solver(grid_initial)
    t1 = time.time()
    solution_gid = solver.solve()
    t1 = time.time()-t1
    t2 = time.time() 
    solved = solver.is_sudoku_solved_2()
    t2 = time.time()-t2
    print(f"Solution Grid:\n{sudoku_grid_to_string(solution_gid)}")
    print(f"The solution is {'correct' if solved else 'incorrect'}!")
    print(f"Time to find the solution: {round(t1,3)}")
    print(f"Time to check the solution: {round(t2,3)}")

    if solver.unsolved_cells:
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

    parser.add_argument('quiz_file', help=".xlsx file that has a SUDOKU quiz")  # positional argument
    parser.add_argument('--log', action='store_true', help="Enables logging to file")    # on/off flag
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
