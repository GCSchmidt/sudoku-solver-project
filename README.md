# SUDOKU SOLVER

This project contains a Python-based program developed to solve Sudoku quizzes. The primary objective of the program is to solve all quizzes from [this Sudoku dataset from Kaggle](https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download). The program provides the ability to solve Sudoku quizzes from `.xlsx` files or images of the Sudoku quiz taken from [sudoku.com](https://sudoku.com/). Images must contain only the Sudoku quiz for the program to work correctly with the image input. The file at `./inputs/image/extreme.png` is an example of such a quiz. The program has successfully solved quizzes of the most challenging difficulty, which is **Extreme**. A simple Web App version of the program was also made.

## SETUP

After cloning repo, run the following to setup everything for the code to work. This is assuming python 3.10 is installed:

- For Windows:
  
    ```cmd
    python scripts\\setup.py && .venv\\Scripts\\activate
    ```

- For Linux:
  
    ``` bash
    python3 scripts/setup.py && source .venv/bin/activate
    ```

## Try Solving a Single SUDOKU

The `sudoku_solver.py` program offers the ability to solve Sudoku quizzes from `.xlsx` files or image files (`.png`, `.jpg`). Some examples of input files can be found under the  `.\inputs\` folder.

To get started, try running the program to solve a quiz in .xlsx format using the following command:

``` bash
python3 scripts/sudoku_solver.py ./inputs/xlsx/intermediate.xlsx
```

Or Try and solve a quiz from an image with one of the examples using the following command:

``` bash
python3 scripts/sudoku_solver.py ./inputs/images/quiz_3.png
```

You can modify the `./inputs/xlsx/problem_template.xlsx` accordingly to create a new quiz for the program. Alternatively, visit [sudoku.com](https://sudoku.com/), take a screenshot of a Sudoku grid, and try solving it with the program.

## Run The Simple Web App
To launch the simple web app version of the program run the following:

``` bash
python3 web_app/app.py 
```

And then go to [`http://127.0.0.1:5000`](http://127.0.0.1:5000) in your browser

## Try Solving a Thousand Quizzes

This `./datasets/sudoku.csv` contains a csv file with 1 Million Sudoku quizzes. Try and solve 1 thousand of these by running:

``` bash
python3 scripts/test.py TestSolver.test_1k_quizzes
```

Hopefully, you will see:

*From 1000 total quizzes, 1000 were correctly solved!*

To verify the results, you can compare the expected and actual solutions in the CSV file located at `./output/test_1k_results.csv`

# Notebooks
The directory `./notebooks/` contains 3 notebooks:
1. For solving all 1 million quizzes from [this dataset](https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download)
2. Training a digit classifier
3. Detailing the process of solving sudoku quizzes from image

## Future Ideas

1. Get the program to work for images of printed sudoku quizzes, e.g those found in magazines or newspapers.
2. Speed up 'image-to-quiz' pre-processing by filtering out corner features more efficiently.
