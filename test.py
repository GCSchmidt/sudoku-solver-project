import unittest
import os
import time
import random
import pandas as pd
import sudoku_solver as ss
from sudoku_solver import Solver
import multiprocessing as mp
from functools import partial


QUIZ_DF = pd.read_csv(r'datasets/sudoku.csv')


def solve_quiz(quiz_num):
    """
    Helper function to solve a single quiz.
    """
    global QUIZ_DF
    # load quiz and solution
    quiz, expected_solution = ss.load_quiz_from_dataset(QUIZ_DF, quiz_num)
    solver = Solver(quiz)
    solver.try_hard = True
    actual_solution_array = solver.solve()
    actual_solution_str = "".join([str(num) for num in actual_solution_array.flatten()])
    quiz_result = (expected_solution == actual_solution_str)
    return quiz_num, expected_solution, actual_solution_str, quiz_result


class TestSolver(unittest.TestCase):

    def test_single_quiz(self):
        print("\nRunning single quiz test...")
        results_df = pd.DataFrame(columns=['id', 'expected_solution', 'actual_solution', 'correct'])
        quiz_num = 1_000
        _, _, _, quiz_result = solve_quiz(quiz_num)
        results_df.to_csv(r'outputs/test_single_results.csv', index=False)
        print(f"\tThe solution of quiz {quiz_num} is {'correct' if quiz_result else 'incorrect'}!")
        self.assertTrue(quiz_result)
  
    def test_1k_quizzes(self):
        global QUIZ_DF

        print("Running 1k quizzes test...")
        passed_n = 0
        N = 1_000
        quiz_nums = random.sample(range(0, 1_000), N)
        timer = time.time()
        # Use multiprocessing to solve quizzes in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            solve_quiz_partial = partial(solve_quiz)
            results = pool.map(solve_quiz_partial, quiz_nums)

        # See if there are any failures
        for _, _, _, quiz_result in results:
            if quiz_result:
                passed_n += 1

        results_df = pd.DataFrame(results, columns=['id', 'expected_solution', 'actual_solution', 'correct'])

        timer = time.time() - timer
        os.makedirs('outputs', exist_ok=True)
        results_df.to_csv(r'outputs/test_1k_results.csv', index=False)
        print(f"\tIt took {round(timer, 3)} seconds to solve {N} SUDOKU quizzes.")
        print(f"\tFrom {N} total quizzes, {passed_n} were correctly solved!")
        self.assertTrue(passed_n == N)


if __name__ == '__main__':
    print("-"*70, "\nRunning Tests...")
    unittest.main()
