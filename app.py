
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load saved AI model and TF-IDF vectorizer
model = joblib.load("model/news_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        news_text = data.get("text", "").strip()

        if not news_text:
            return jsonify({"error": "Please enter some news text."}), 400

        # Convert input text into TF-IDF features
        text_vector = tfidf.transform([news_text])

        # Predict news category
        prediction = model.predict(text_vector)[0]

        return jsonify({
            "category": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
