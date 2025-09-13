import numpy as np
import json
import pygame
from pygame.locals import *
from PIL import Image, ImageOps

# Load weights and biases
with open('hidden_layer_data.json', 'r') as f:
    hidden_layer_data = json.load(f)
with open('output_layer_data.json', 'r') as f:
    output_layer_data = json.load(f)

def relu(x):
    return np.maximum(0, x)

def predict_digit(image):
    flattened_image = image.flatten()
    
    hidden_out = np.zeros(10)
    for i in range(10):
        weights = hidden_layer_data[i]["weights"]
        bias = hidden_layer_data[i]["bias"]
        weight = np.dot(weights, flattened_image) + bias
        hidden_out[i] = relu(weight)

    output_out = np.zeros(10)
    for i in range(10):
        weights = output_layer_data[i]["weights"]
        bias = output_layer_data[i]["bias"]
        weight = np.dot(weights, hidden_out) + bias
        output_out[i] = weight

    exp_output = np.exp(output_out - np.max(output_out))
    softmax_output = exp_output / exp_output.sum()

    return softmax_output

def prepare_image(image):
    # Resize to 28x28 while maintaining aspect ratio
    aspect_ratio = image.size[0] / image.size[1]
    if aspect_ratio > 1:
        new_size = (28, int(28 / aspect_ratio))
    else:
        new_size = (int(28 * aspect_ratio), 28)
    
    image = image.resize(new_size, Image.LANCZOS)
    print(image)
    # Create a new 28x28 image and paste the resized image on it
    new_image = Image.new('L', (28, 28), 0)  # 'L' for grayscale mode, 0 for black background
    paste_pos = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    new_image.paste(image, paste_pos)

    # Convert to numpy array and apply thresholding
    new_image = np.array(new_image)
    new_image = new_image.astype('float32') / 255
    new_image = np.where(new_image >= 0.5, 1, 0).astype('int8')
    print(new_image)
    return new_image


# Initialize pygame
pygame.init()

# Set up display
width, height = 200, 200
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Draw a Digit")

# Set up drawing variables
drawing = False
last_pos = None
color = (255, 255, 255)
radius = 10

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            drawing = True
            last_pos = event.pos
        elif event.type == MOUSEBUTTONUP:
            drawing = False
        elif event.type == MOUSEMOTION:
            if drawing:
                pygame.draw.line(window, color, last_pos, event.pos, radius)
                last_pos = event.pos
        elif event.type == KEYDOWN:
            if event.key == K_c:
                window.fill((0, 0, 0))
            elif event.key == K_p:
                pygame.image.save(window, "drawing.png")
                img = Image.open("drawing.png")
                img = prepare_image(img)
                prediction = predict_digit(img)
                digit = np.argmax(prediction)
                print(f'Predicted Digit: {digit}')
                print(f'Softmax Output: {prediction}')
    
    pygame.display.flip()

pygame.quit()

'''
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 '''