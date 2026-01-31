import os
import time
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sudoku_solver as ss
from sudoku_solver import Solver


parent_dir = os.path.dirname(os.path.dirname(__file__))
dataset_dir = os.path.join(parent_dir, "datasets")
output_dir = os.path.join(parent_dir, "outputs")
QUIZ_DF = pd.read_csv(os.path.join(dataset_dir, "sudoku.csv"))


def solve_quiz(quiz_num, mode):
    """
    Helper function to solve a single quiz.
    """
    global QUIZ_DF
    # load quiz and solution
    quiz, expected_solution = ss.load_quiz_from_dataset(QUIZ_DF, quiz_num)
    solver = Solver(quiz)
    solver.mode = mode
    start = time.perf_counter()
    actual_solution_array = solver.solve()
    elapsed = time.perf_counter() - start
    actual_solution_str = "".join([str(num) for num in actual_solution_array.flatten()])
    quiz_result = (expected_solution == actual_solution_str)
    return quiz_num, expected_solution, actual_solution_str, quiz_result, elapsed


def main():
    N = 10_000
    quiz_nums = random.sample(range(0, 1_000_000), N)
    df = run_brute_force(quiz_nums)
    df = pd.concat([df, run_try_hard(quiz_nums)])
    df = pd.concat([df, run_simple(quiz_nums)])
    boxplot = sns.boxplot(data=df, x="mode", y="time", hue="mode")
    boxplot_figure = boxplot.get_figure()
    boxplot_figure.savefig(os.path.join(output_dir, "time_boxplot.png"))
    displot = sns.displot(data=df, x="correct", col="mode", hue="mode", bins=[0,1])
    displot_figure = displot.figure
    displot_figure.savefig(os.path.join(output_dir, "correct_plot.png"))
    timer_description = df.groupby(by="mode")['correct'].describe()
    correct_description = df.groupby(by="mode")['time'].describe()
    with open(os.path.join(output_dir, "performace_results.txt"), 'w') as txt_file:
        txt_file.write("Time Description:")
        txt_file.write(timer_description.to_string()+"\n")
        txt_file.write("Correct Description:")
        txt_file.write(correct_description.to_string()+"\n")

def run(quiz_nums, mode):
    results = [solve_quiz(quiz_num, mode) for quiz_num in quiz_nums]
    results_df = pd.DataFrame(results, columns=['id', 'expected_solution', 'actual_solution', 'correct', 'time'])
    return results_df


def run_brute_force(quiz_nums):
    mode = ss.Modes.BRUTE_FORCE
    results_df = run(quiz_nums, mode)
    results_df["mode"] = "BRUTE_FORCE"
    return results_df


def run_try_hard(quiz_nums):
    mode = ss.Modes.TRY_HARD
    results_df = run(quiz_nums, mode)
    results_df["mode"] = "TRY_HARD"
    return results_df


def run_simple(quiz_nums):
    mode = ss.Modes.SIMPLE
    results_df = run(quiz_nums, mode)
    results_df["mode"] = "SIMPLE"

    return results_df


if __name__ == '__main__':
    main()