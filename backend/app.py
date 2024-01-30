from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import get_rag_answer, create_general_llm

app = Flask(__name__)
CORS(app)
user_query_list = []
general_llm = create_general_llm()


@app.post("/predict")
def predict():
    user_query = request.get_json().get("message")
    user_query_list.append(user_query)
    # Check if text is valid
    answer = get_rag_answer(user_query_list, general_llm)
    response = {"answer": answer}
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
