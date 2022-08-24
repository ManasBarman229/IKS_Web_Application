
from flask import Flask, redirect, url_for, render_template, request
import model as m

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/manual')
def manual():
    return render_template('manual.html')


@app.route('/hashtag_query')
def hashtag_query():
    return render_template('hashtag_query.html')


# @app.route('/scrapper_data')
# def scrapper_data():
#     return render_template('scrapper_data.html', headings=headings, data=data)

# manual data prediction


@ app.route('/manual_query_result', methods=['POST'])
def submit():
    if request.method == "POST":
        manual_query = request.form["manual_query"]
        result = m.manual_query_input(manual_query)
        print(result)
        n = result
    return render_template("manual.html", prediction_text=n)


headings = ['text', 'output']
# twitter data prediction


@ app.route('/hashtag_query_result', methods=['GET', 'POST'])
def hashtag_query_result():
    if request.method == "POST":
        scrapper_query = request.form["scrapper_query"]
        data = m.getData(scrapper_query)
        return render_template('scrapper_data.html', headings=headings, data=data)

    return render_template('hashtag_query.html')


# main driver function
if __name__ == '__main__':
    app.run(debug=True)
