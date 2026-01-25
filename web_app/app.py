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
    solution_str = "563287712364515449364739413679388783371713317561552517842763311235389421938888645"
    return render_template('solution.html', solution_str=solution_str)


if __name__ == '__main__':
    app.run(debug=True)