import os
import sys
from typing import Tuple
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(1, parent_dir)
import scripts.sudoku_solver as sudoku_solver

ALLOWED_EXTENSIONS = {'png', 'jpeg'}


def allowed_file(filename) -> Tuple[bool, str | None]:
    if '.' in filename:
        file_extension = filename.rsplit('.', 1)[1].lower()
    else:
        return False, None

    return file_extension in ALLOWED_EXTENSIONS, file_extension


def parse_and_validate_grid(cell_values: list[str]) -> Tuple[sudoku_solver.Solver, str]:
    quiz_str = "".join(cell_values)
    initial_grid = sudoku_solver.quiz_str_to_grid(quiz_str)
    solver = sudoku_solver.Solver(initial_grid)
    valid_grid = solver.is_valid_intermediat_grid()
    string_grid = sudoku_solver.sudoku_grid_to_string(initial_grid)
    if not valid_grid:
        return None, "Invalid Sudoku Quiz was Submitted. Try Again."

    return solver, None


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 1 * 1000 * 1000  # 1MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/manual_input', methods=['POST'])
def manual_input():
    error = None
    if request.method == 'POST':
        if 'file' not in request.files:
            error = "No File was Uploaded"
            return render_template('manual_input.html', error=error)
        else:
            file = request.files['file']
            filename = secure_filename(file.filename)
            file_check, file_extension = allowed_file(filename)
            if file_check:
                filename = f"image_file.{file_extension}"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                error = "Invalid File was Uploaded"
                return render_template('manual_input.html', error=error)
        filepath = os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        grid = sudoku_solver.image_to_sudoku_quiz(filepath)
        cell_values = "".join([str(val) for val in grid.flatten()])
        return render_template(
            'manual_input.html',
            error=error,
            preset_cell_values=cell_values
        )
    else:
        error = request.args.get('error')
        return render_template('manual_input.html', error=error)


@app.route('/solution', methods=['POST'])
def solution():
    cell_values = [request.form.get(f"cell_{i}", "0") for i in range(81)]

    solver, error = parse_and_validate_grid(cell_values)

    if error:
        return render_template(
            'manual_input.html',
            error=error,
            preset_cell_values=cell_values
            )

    solution_grid = solver.solve()
    solved = solver.is_sudoku_solved()

    return render_template('solution.html', solution_grid=solution_grid, solved=solved)


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    error = "Uploaded File was too Large"
    return render_template('manual_input.html', error=error)


if __name__ == '__main__':
    app.run(debug=False)
