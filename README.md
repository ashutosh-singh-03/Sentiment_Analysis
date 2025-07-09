# Sentiment Analysis App

A Flask web application that analyzes the sentiment of text using an LSTM neural network model.

## Features

- Real-time sentiment analysis (Positive/Negative)
- Confidence percentage for predictions
- Clean, responsive web interface
- Text preprocessing with NLTK

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Sentiment_Analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Open your browser** and go to `http://127.0.0.1:5000`

## Usage

1. Enter any text in the input field
2. Click "Analyze Sentiment"
3. View the sentiment result and confidence percentage

## Project Structure

```
├── main.py              # Flask application
├── requirements.txt     # Python dependencies
├── models/
│   ├── lstm_model.h5    # Trained LSTM model
│   └── tokenizer.pkl    # Text tokenizer
└── templates/
    └── index.html       # Web interface
```

## Technologies Used

- **Backend**: Flask, TensorFlow/Keras
- **Frontend**: HTML, CSS, Bootstrap
- **NLP**: NLTK, NumPy
- **Model**: LSTM Neural Network

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Flask
- NLTK