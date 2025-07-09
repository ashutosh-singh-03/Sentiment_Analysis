import os
import pickle
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")


# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model("models\lstm_model.h5")
with open("models\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
# Text Preprocessing Function
def clean_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))


    text = text.lower() # lowercasing
    text = re.sub(r"[^a-z\s]", " ", text) # remove characters etc
    words = text.split() # tokenization
    words = [ps.stem(word) for word in words if word not in stop_words] # Stemming

    return " ".join(words)

# Prediction function
def predict_sentiment(text):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    # Predict Sentiment
    prediction = model.predict(padded_sequence)[0][0]

    # check if pos or negative
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    # Convert confidence to percentage
    confidence_percentage = float(prediction) * 100 if prediction > 0.5 else (1 - float(prediction)) * 100
    return sentiment, f"{confidence_percentage:.1f}%"

# routes
@app.route("/", methods = ['GET','POST'])
def home():
    result = None
    confidence = None
    if request.method == 'POST':
        text = request.form.get("text")
        if text:
            result, confidence = predict_sentiment(text)
    return render_template("index.html", result = result, confidence = confidence)
# Python Main
if __name__ == "__main__":
    app.run(debug=True)