# AI-Powered News Classification System

## Project Overview

An AI-powered web application that automatically classifies news articles into 7 different categories using Machine Learning and Natural Language Processing (NLP).

Users can enter news text through a simple web interface and receive a predicted news category.

## Technologies Used

- Python
- Flask
- Scikit-learn
- TF-IDF Vectorization
- Linear Support Vector Classifier (LinearSVC)
- HTML, CSS, JavaScript
- Pandas, NumPy
- Matplotlib, Seaborn

## Machine Learning Workflow

1. Dataset inspection and preprocessing
2. Handling class imbalance using class weights and oversampling
3. Text feature extraction using TF-IDF
4. Model training using LinearSVC
5. Model evaluation using accuracy, precision, recall and F1-score
6. Confusion matrix and performance visualization

## Model Performance

- Accuracy: 89.50%
- Macro F1-score: 89.25%
- Number of categories: 7

## Project Structure

```text
News Classification/
├── app.py
├── inspect_data.py
├── preprocess.py
├── train_model.py
├── evaluate_model.py
├── compare_models.py
├── visualize_results.py
├── save_model.py
├── model/
│   ├── news_model.pkl
│   └── tfidf_vectorizer.pkl
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run the Project

1. Clone or download this repository.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Start the Flask application:

```bash
python app.py
```

4. Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Note

The original dataset is not included in this repository because of its large file size. The trained model and TF-IDF vectorizer are included.
