from flask import Flask, request, jsonify
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences 
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Initialize the tokenizer and train it on the training data
tokenizer = Tokenizer(num_words=5000, lower=True, split=' ')
X_train = ["I love this!", "This is terrible...", "I am happy today!"]  # Sample training data
tokenizer.fit_on_texts(X_train)

# Load the trained model
model = tf.keras.models.load_model('C:/Users/naidu/Documents/UTD/Projects/Social Media Sentiment Analysis/.venv/sentiment_model.h5')



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    # Tokenize and pad the input text
    text_seq = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_seq, maxlen=100)
    # Predict sentiment
    prediction = model.predict(text_padded)
    sentiment = "positive" if prediction[0] > 0.5 else "negative"

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
