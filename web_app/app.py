from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/manual_input')
def manual_input():
    return render_template('manual_input.html')

@app.route('/solution')
def solution():
    solution_arr = [[i for i in range(1, 10)]] * 9  #
    return render_template('solution.html', solution_arr=solution_arr)


if __name__ == '__main__':
    app.run(debug=True)