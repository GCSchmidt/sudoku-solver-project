import os
import sys
from typing import Tuple
from flask import Flask, render_template, flash, redirect, url_for, request

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(1, parent_dir)
import scripts.sudoku_solver as sudoku_solver


def parse_and_validate_grid(cell_values: list[str]) -> Tuple[sudoku_solver.Solver, str]:
    quiz_str = "".join(cell_values)
    initial_grid = sudoku_solver.quiz_str_to_grid(quiz_str)
    solver = sudoku_solver.Solver(initial_grid)
    valid_grid = solver.is_valid_intermediat_grid()
    string_grid = sudoku_solver.sudoku_grid_to_string(initial_grid)
    print(string_grid)
    if not valid_grid:
        return None, "Invalid Sudoku Quiz was Submitted. Try Again."

    return solver, None


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/manual_input')
def manual_input():
    error = request.args.get('error')
    return render_template('manual_input.html', error=error)


@app.route('/solution', methods=['POST'])
def solution_post():
    cell_values = [request.form.get(f"cell_{i}", "0") for i in range(81)]

    solver, error = parse_and_validate_grid(cell_values)

    if error:
        return render_template(
            'manual_input.html',
            error=error,
            )
    
    cell_grid = solver.grid_intermediate
    
    string_grid = sudoku_solver.sudoku_grid_to_string(cell_grid)
    print(string_grid)

    solution_grid = solver.solve()
    solved = solver.is_sudoku_solved()

    return render_template('solution.html', solution_grid=solution_grid, solved=solved)

if __name__ == '__main__':
    app.run(debug=True)
