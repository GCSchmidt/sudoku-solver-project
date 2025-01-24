# SUDOKU SOLVER

This is a Python program designed to solve Sudoku quizzes with difficulty levels up to "hard." It provides a reliable solution for most standard Sudoku quizzes. The goal of the program was to successfully solve all the quizzes in this [dataset](https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download).

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
py ./test.py
```

Hopefully, you should see:

*From 1000 total quizzes, 1000 were correctly solved!*

To verify the results, you can compare the expected and actual solutions in the CSV file located at `./output/test_results.csv`

## To Generate Quizzes

Sudoku quizzes can be generated with [this](https://www.ocf.berkeley.edu/~arel/sudoku/main.html).

## Future Plans

1. Develop a method to solve sudoku quizes from screenshots.
2. Solve quized of 'evil' difficulty.
