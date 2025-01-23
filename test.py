import pandas as pd
import unittest
import time
import random
import sudoku_solver as ss
from sudoku_solver import Solver
import multiprocessing as mp
from functools import partial
import os

QUIZ_DF = pd.read_csv(r'datasets/sudoku.csv')


def solve_quiz(quiz_num):
    """
    Helper function to solve a single quiz.
    """
    # load quiz and solution
    global QUIZ_DF

    quiz, expected_solution = ss.load_quiz_from_dataset(QUIZ_DF, quiz_num)
    solver = Solver(quiz)
    solver.try_hard = False
    actual_solution_array = solver.solve()
    actual_solution_str = "".join([str(num) for num in actual_solution_array.flatten()])
    quiz_result = (expected_solution == actual_solution_str)
    return quiz_num, expected_solution, actual_solution_str, quiz_result


class TestSolver(unittest.TestCase):

    def test_1k_quizes(self):
        global QUIZ_DF
        resuls_df = pd.DataFrame(columns=['id', 'expected_solution', 'actual_solution', 'correct'])
        passed_n = 0
        N = 1_000                    # len(QUIZ_DF)
        quiz_nums = random.sample(range(0, 1_000_000), N) 
        timer = time.time()
        # Use multiprocessing to solve quizzes in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            solve_quiz_partial = partial(solve_quiz)
            results = pool.map(solve_quiz_partial, quiz_nums)

        # Process results
        for quiz_num, expected_solution, actual_solution_str, quiz_result in results:
            resuls_df.loc[quiz_num] = [quiz_num, expected_solution, actual_solution_str, quiz_result]
            if quiz_result:
                passed_n += 1

        timer = time.time() - timer
        os.makedirs('outputs', exist_ok=True)
        resuls_df.to_csv(r'outputs/test_results.csv', index=False)
        print(f"It took {round(timer, 3)} to solve {N} SUDOKU quizzes.")
        print(f"From {N} total quizzes, {passed_n} were correctly solved!")
        self.assertTrue(passed_n == N)


if __name__ == '__main__':
    print("-"*70, "\nRunning Tests...")
    unittest.main()
