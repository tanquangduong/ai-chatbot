from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response

app = Flask(__name__)
CORS(app)

# @app.get('/')
# def index_get():
#     return render_template('base.html')

@app.post('/predict')
def predict():
    text = request.get_json().get("message")
    # |TODO|: Check if text is valid
    answer = get_response(text)
    response = {"answer": answer}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)