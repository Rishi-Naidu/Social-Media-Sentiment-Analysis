import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Prepare data
X_train = ["I love this!", "This is terrible...", "I am happy today!"]  # Sample training data
y_train = [1, 0, 1]  # 1 = Positive, 0 = Negative

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=5000, lower=True, split=' ')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_seq, maxlen=100)
    
# Convert to NumPy array and specify data type
X_train_padded = np.array(X_train_padded, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 128, input_length=X_train_padded.shape[1]),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_padded, y_train, epochs=20, batch_size=32)

# Save the trained model
model.save('sentiment_model.h5')
print("Model saved as 'sentiment_model.h5'.")
