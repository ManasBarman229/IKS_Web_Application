
from glob import glob
from flask import Flask, redirect, url_for, render_template, request
import model as m

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/manual')
def manual():
    return render_template('manual.html')


@ app.route('/manual_query_result', methods=['POST'])
def submit():
    if request.method == "POST":
        manual_query = request.form["manual_query"]
        result = m.manual_query_input(manual_query)
        print(result)
        n = result
    return render_template("manual.html", prediction_text=n)


# main driver function
if __name__ == '__main__':
    app.run(debug=True)
