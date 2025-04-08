import tensorflow as tf
import numpy as np
import requests
from bs4 import BeautifulSoup

# Get the text of the book (e.g., Pride and Prejudice)
url = "https://www.gutenberg.org/files/1342/1342-0.txt"  # Pride and Prejudice
response = requests.get(url)
text = response.text

# Preprocess text (cleaning it up and removing non-alphabetic characters)
text = text.lower()
text = ''.join([char if char.isalpha() or char.isspace() else ' ' for char in text])

# Use only the first 1000 characters for training
text = text[:10000]

# Create tokenization
chars = sorted(set(text))
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}


text_as_int = np.array([char_to_index[char] for char in text])

# Create training sequences
seq_length = 100  # Sequence length
x_data = []
y_data = []

for i in range(len(text_as_int) - seq_length):
    x_data.append(text_as_int[i:i + seq_length])
    y_data.append(text_as_int[i + seq_length])

x_data = np.array(x_data)
y_data = np.array(y_data)

# Define the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(chars), output_dim=256, input_length=seq_length),
    tf.keras.layers.LSTM(512, return_sequences=False),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model for 10 epochs
model.fit(x_data, y_data, epochs=10)


# Define text generation function with temperature and top-k sampling
def sample_with_temperature(predictions, temperature=1.0, top_k=10):
    predictions = predictions[0]
    predictions = predictions / temperature
    predictions = tf.nn.softmax(predictions)
    predictions = predictions.numpy()

    if top_k is not None:
        top_k_indices = predictions.argsort()[-top_k:][::-1]
        top_k_probs = predictions[top_k_indices]
        selected_index = np.random.choice(top_k_indices, p=top_k_probs / top_k_probs.sum())
    else:
        selected_index = np.random.choice(len(predictions), p=predictions)

    return selected_index


# Preprocess the start string to match the model's preprocessing
def preprocess_start_string(start_string):
    start_string = start_string.lower()
    return ''.join([char if char in char_to_index else ' ' for char in start_string])


def generate_text(start_string, num_generate=500, temperature=1.0, top_k=10):
    start_string = preprocess_start_string(start_string)
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    generated_text = start_string

    for _ in range(num_generate):
        predictions = model(input_eval)

        predicted_id = sample_with_temperature(predictions, temperature=temperature, top_k=top_k)

        generated_text += index_to_char[predicted_id]
        input_eval = tf.expand_dims([predicted_id], 0)

    return generated_text


# Example : Generate text based on a starting string
start_string = "it is a truth universally acknowledged"
generated_text = generate_text(start_string, num_generate=500, temperature=0.8, top_k=10)

# Print the generated story
print(generated_text)
