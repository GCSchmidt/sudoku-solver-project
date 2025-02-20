# SUDOKU SOLVER

This project is a Python-based solution developed to reliably solve Sudoku quizzes. The primary objective of the program is to solve all quizzes in the Sudoku dataset from [Kaggle](https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download). The solver provides the ability to solve Sudoku quizzes from `.xlsx` files or images of the Sudoku grids taken from [sudoku.com](https://sudoku.com/). Images must contain only the Sudoku grid for the solver to work correctly with the image input. The solver is currently unable to solve the quizzes catagorized as **Evil** (the level above **Hard**). The file at `./inputs/xlsx/evil.xlsx` is an example of such a quiz.

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

The `sudoku_solver.py` program offers the ability to solve Sudoku quizzes from `.xlsx` files or image files (`.png`, `.jpg`). Some examples of input files can be found under the  `.\inputs\` folder.

To get started, try running the solver to solve a quiz in .xlsx format using the following command:

``` bash
py ./sudoku_solver.py ./inputs/xlsx/intermediate.xlsx
```

Or Try and solve a quiz from an image with one of the examples using the following command:

``` bash
py ./sudoku_solver.py ./inputs/images/quiz_3.png
```

You can modify the `./inputs/xlsx/problem_template.xlsx` accordingly to create a new quiz for the solver. Alternatively, visit [sudoku.com](https://sudoku.com/), take a screenshot of a Sudoku grid, and try solving it with the program..

## Try Solving a Thousand Quizzes

This `./datasets/sudoku.csv` contains a csv file with 1 Million Sudoku quizzes. Try and solve 1 thousand of these by running:

``` bash
py ./test.py TestSolver.test_1k_quizzes
```

Hopefully, you will see:

*From 1000 total quizzes, 1000 were correctly solved!*

To verify the results, you can compare the expected and actual solutions in the CSV file located at `./output/test_1k_results.csv`

## Solving 1 Million Quizzes

The `sudoku_solver_solve_them_all.ipynb` jupyter notebook was used to solve all the quizzes in the dataset. All the quizzes were successfully solved!

## To Generate Quizzes

Sudoku quizzes can be generated with [this](https://www.ocf.berkeley.edu/~arel/sudoku/main.html).

## Future Plans

1. Solve quizzes of 'evil' difficulty.
2. Speed up 'image-to-quiz' pre-processing by filtering out corner features more efficiently.

## Extra

### Training a Digit Classifier

To enable the solver to process Sudoku quizzes from images, a digit recognition model was trained to identify digits from images. The training process is documented in the `digit_classifier.ipynb` notebook, which outlines the steps taken to develop and train the model.  

### Solving Sudoku Quizes from Images

The process of solving a Sudoku quiz from an image is documented in the `sudoku_image_solver.ipynb` notebook. The notebook outlines the steps involved to process solve a Sudoku quiz from an image.
