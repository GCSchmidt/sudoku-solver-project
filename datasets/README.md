# Datasets

## Puzzle Dataset

`sudoku.zip` contains a single file called `sudoku.csv`. The file contains 1 million Sudoku puzzles along with their solutions. It was downloaded from [this Kaggle dataset](https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download).

The file `sudoku.csv` has two columns:

1. `quizzes` – a string of 81 digits representing the puzzle
2. `solutions` – a string of 81 digits representing the solution

## Digit Dataset for Training a Digit Classifier

The file `digit_templates.zip` contains one .png image for each digit (1–9), as well as a blank cell (`0.png`). These images were captured from [sudoku.com](https://sudoku.com/).

The images in `digit_templates.zip` are used in the `digit_classifier.ipynb` notebook to generate a dataset called `digit_dataset.zip`. This dataset contains 10 directories — one for each digit — each with 100 augmented versions of the corresponding digit.

The resulting dataset is then used to train a model capable of classifying digits in the `digit_classifier.ipynb` notebook.
