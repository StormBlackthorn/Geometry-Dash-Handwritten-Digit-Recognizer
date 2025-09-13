import numpy as np
import matplotlib.pyplot as plt
import json
import keras
from keras.datasets import mnist

# Load weights and biases
with open('hidden_layer_data.json', 'r') as f:
    hidden_layer_data = json.load(f)
with open('output_layer_data.json', 'r') as f:
    output_layer_data = json.load(f)

def relu(x):
    return np.maximum(0, x)

def predict_digit(image):
    # Flatten the image
    flattened_image = image.flatten()

    # Calculate hidden layer output
    hidden_out = np.zeros(10)
    for i in range(10):
        weights = hidden_layer_data[i]["weights"]
        bias = hidden_layer_data[i]["bias"]
        weight = np.dot(weights, flattened_image) + bias
        hidden_out[i] = relu(weight)  # Applying ReLU activation function

    # Calculate output layer output
    output_out = np.zeros(10)
    for i in range(10):
        weights = output_layer_data[i]["weights"]
        bias = output_layer_data[i]["bias"]
        weight = np.dot(weights, hidden_out) + bias
        output_out[i] = weight

    # Calculate softmax
    exp_output = np.exp(output_out - np.max(output_out))
    softmax_output = exp_output / exp_output.sum()

    return softmax_output

def clean_data(X_test, Y_test):
    X_test = X_test.reshape(10000, 784).astype('float32') / 255
    X_test = np.where(X_test >= 0.5, 1, 0).astype('int8')  # Binary values

    Y_test = keras.utils.to_categorical(Y_test, 10)  # 1-hot encoded

    return X_test, Y_test

def forward(iterations=10):
    (_, _), (X_test, Y_test) = mnist.load_data()
    X_test, Y_test = clean_data(X_test, Y_test)

    count = 0
    total = 0

    for X, y in zip(X_test, Y_test):
        prediction = predict_digit(X)

        if np.argmax(prediction) == np.argmax(y):
            count += 1
        total += 1

        if total >= iterations:
            break

        if total % 100 == 0:
            print(total)

    print(f'Accuracy: {count} / {total} = {count / total * 100:.2f}%')

def main():
    # Forward function with sample data from mnist dataset
    forward(iterations=10000)

if __name__ == '__main__':
    main()
