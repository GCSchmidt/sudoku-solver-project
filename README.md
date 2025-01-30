# SUDOKU SOLVER

This project is a Python-based solution developed to reliably solve Sudoku puzzles. The primary objective of the program is to solve all quizzes in the Sudoku dataset from [Kaggle](https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download).

## SETUP

After cloning repo, run the following to setup everything for the code to work. This is assuming python 3.10 is installed:

- For Windows:
  
    ```cmd
    python setup.py && .venv\\Scripts\\activate
    ```

- For Linux:
  
    ``` bash
    python setup.py && source .venv/bin/activate
    ```

## Try Solving a Single SUDOKU

In the `./inputs` folder, you'll find some example sudoku quiz files. To get started, try running the solver with the following command:

``` bash
py ./sudoku_solver.py ./inputs/intermediate.xlsx
```

You can modify the `./inputs/problem_template.xlsx` accordingly to create a new quiz for the solver.

The solver is currently unable to solve the `./inputs/evil.xlsx` quiz.

## Try Solving a Thousand Quizzes

This `./datasets/sudoku.csv` contains a csv file with 1 Million Sudoku quizzes. Try and solve 1 thousand of these by running:

``` bash
py ./test.py TestSolver.test_1k_quizes
```

Hopefully, you will see:

*From 1000 total quizzes, 1000 were correctly solved!*

To verify the results, you can compare the expected and actual solutions in the CSV file located at `./output/test_1k_results.csv`

## Solving 1 Million Quizzes

The `sudoku_solver_solve_them_all.ipynb` jupyter notebook was used to solve all the quizzes in the dataset. All the quizzes were successfully solved!

## To Generate Quizzes

Sudoku quizzes can be generated with [this](https://www.ocf.berkeley.edu/~arel/sudoku/main.html).

## Future Plans

1. Develop a method to solve Sudoku quizes from screenshots.
2. Solve quizzes of 'evil' difficulty.
