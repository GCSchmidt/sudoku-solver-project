import os
import sys
from flask import Flask, render_template, request

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(1, parent_dir)
import scripts.sudoku_solver as sudoku_solver

SOLVER: sudoku_solver.Solver

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/manual_input')
def manual_input():
    return render_template('manual_input.html')


@app.route('/solution', methods=['POST'])
def solution():
    cell_values = []
    for i in range(81):
        cell_values.append(request.form.get(f"cell_{i}", "0"))
    quiz_str = "".join(cell_values)
    initial_grid = sudoku_solver.quiz_str_to_grid(quiz_str)
    SOLVER = sudoku_solver.Solver(initial_grid)
    solution_gid = SOLVER.solve()
    solved = SOLVER.is_sudoku_solved()
    #  solution_arr = [[i for i in range(1, 10)]] * 9
    return render_template(
        'solution.html',
        solution_grid=solution_gid,
        solved=solved
        )


if __name__ == '__main__':
    app.run(debug=True)
