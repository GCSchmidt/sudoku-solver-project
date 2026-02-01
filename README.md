# SUDOKU SOLVER

This project contains a Python-based program developed to solve Sudoku quizzes. The primary objective of the program is to solve all quizzes from [this Sudoku dataset from Kaggle](https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download). The script at `scripts/sudoku_solver.py` is a program which has the logic implemented to solve Sudoku quizzes from `.xlsx` files or images of the Sudoku quiz taken from [sudoku.com](https://sudoku.com/). Images must contain only the Sudoku quiz for the program to work correctly with the image input. The file at `./inputs/image/extreme.png` is an example of such a quiz. The program has successfully solved quizzes of the most challenging difficulty, which is Extreme. A simple web app version of the program was also made.

## SETUP

After cloning repo, run the following to setup everything for the web app to work.

``` bash
docker build -t ss_app .
docker run -p 5000:5000 ss_app:latest
```

Then go to [`http://127.0.0.1:5000`](http://127.0.0.1:5000) in your browser. You can now visit [sudoku.com](https://sudoku.com/), take a screenshot of a Sudoku grid, and try solving it with the web app or input a Sudoku quiz manually.

# Notebooks
The directory `./notebooks/` contains 3 notebooks:
1. For solving all 1 million quizzes from [this dataset](https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download)
2. Training a digit classifier
3. Detailing the process of solving sudoku quizzes from image

## Future Ideas

1. Get the program to work for images of printed sudoku quizzes, e.g those found in magazines or newspapers.
2. Speed up 'image-to-quiz' pre-processing by filtering out corner features more efficiently.
