import tensorflow as tf
from flask import Flask, render_template, request
from helper import translation

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    english = request.form['english']
    output = translation(english)
    output = output[1:-1]
    output = " ".join(output)
    return render_template('index.html', value=output)
    
if __name__=="__main__":
    app.run(debug=True)