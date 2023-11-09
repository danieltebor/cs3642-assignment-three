import pprint

import numpy as np
import tkinter as tk

from binary_perceptron import BinaryPerceptron
from letter_patterns import *

EMPTY_GRID = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
])

# Place a pattern randomly on a grid.
def randomly_place_pattern_on_grid(grid, pattern):
    # Get the height and width of the grid and pattern.
    grid_height, grid_width = grid.shape
    pattern_height, pattern_width = pattern.shape

    # Get the maximum x and y values for the pattern.
    max_x = grid_width - pattern_width
    max_y = grid_height - pattern_height

    # Get a random x and y value for the pattern.
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    # Place the pattern at the random x and y values.
    grid[y:y + pattern_height, x:x + pattern_width] = pattern

# Randomly place each pattern for each letter on a different grid.
def generate_training_data() -> (np.ndarray, np.ndarray):
    images = np.empty((len(A_PATTERNS) + len(B_PATTERNS),
                       EMPTY_GRID.shape[0] * EMPTY_GRID.shape[1]),
                       dtype=np.float32)
    labels = np.empty(len(A_PATTERNS) + len(B_PATTERNS), dtype=np.int8)

    # Place the patterns for the letter A in a random location for each image.
    for i, pattern in enumerate(A_PATTERNS):
        grid = np.copy(EMPTY_GRID)
        pattern = np.array(pattern)
        randomly_place_pattern_on_grid(grid, pattern)
        grid = grid.flatten()
        images[i] = grid
        labels[i] = 1

    # Place the patterns for the letter B in a random location for each image.
    for i, pattern in enumerate(B_PATTERNS):
        grid = np.copy(EMPTY_GRID)
        pattern = np.array(pattern)
        randomly_place_pattern_on_grid(grid, pattern)
        grid = grid.flatten()
        images[i + len(A_PATTERNS)] = grid
        labels[i + len(A_PATTERNS)] = 0

    return images, labels

def generate_grid_with_random_pattern() -> np.ndarray:
    # Randomly select a pattern for the letter A or B.
    pattern_class = np.random.randint(0, 2)
    if pattern_class == 0:
        pattern_class = A_PATTERNS
    else:
        pattern_class = B_PATTERNS
    pattern = np.random.randint(0, len(pattern_class))
    pattern = np.array(pattern_class[pattern])

    grid = np.copy(EMPTY_GRID)
    randomly_place_pattern_on_grid(grid, pattern)

    return grid

# Create window.                
window = tk.Tk()
window.title('Problem 2')
window.geometry('500x600')

training_metadata_table_frame = tk.Frame(window)
training_metadata_table_frame.grid(row=0, column=0)
training_metadata_table = []

# Train the perceptron.
perceptron = BinaryPerceptron(input_size=EMPTY_GRID.shape[0] * EMPTY_GRID.shape[1], threshold=16, learning_rate=0.01)

def format_weights(weights: np.ndarray, max_weights_to_show: int = 5):
    formatted_weights = '['
    if len(weights) <= max_weights_to_show:
        for weight in weights:
            formatted_weights += f'{weight:.2f}, '
    else:
        for weight in weights[:max_weights_to_show]:
            formatted_weights += f'{weight:.2f}, '
        formatted_weights += '..., '
        formatted_weights += f'{weights[-1]:.2f}, '
    formatted_weights = formatted_weights[:-2] + ']'
    return formatted_weights

training_metadata_table.append('Epoch')
training_metadata_table.append('Weights')
training_metadata_table.append('Avg Loss')
training_metadata_table.append('-')
training_metadata_table.append(format_weights(perceptron.weights))
images, _ = generate_training_data()
avg_loss = 0
for input in images:
    avg_loss += 1 - perceptron.predict(input)
avg_loss /= len(images)
training_metadata_table.append(avg_loss)

MAX_EPOCHS = 100000
num_epochs = 0

# Iterate over a number of epochs.
for i in range(1, MAX_EPOCHS):
    inputs, labels = generate_training_data()
    avg_loss = 0

    # Train the perceptron.
    for input, label in zip(inputs, labels):
        avg_loss += 1 - perceptron.train(input=input, expected_output=label)
        
    avg_loss /= len(inputs)

    if i % 500 == 0:
        training_metadata_table.append(i)
        training_metadata_table.append(format_weights(perceptron.weights))
        training_metadata_table.append(f'{avg_loss:.4f}')

    # Stop training if the perceptron has converged.
    # Convergence is tested by passing in many validation inputs.
    if avg_loss == 0:
        loss_is_greater_than_zero = False
        num_imgs_tested = 0

        for _ in range(16666):
            inputs, labels = generate_training_data()

            for input, label in zip(inputs, labels):
                output = perceptron.predict(input=input)
                num_imgs_tested += 1
                if output != label:
                    loss_is_greater_than_zero = True
                    break
            if loss_is_greater_than_zero:
                break

        if not loss_is_greater_than_zero:
            num_epochs = i
            break

training_metadata_table.append(i)
training_metadata_table.append(format_weights(perceptron.weights))
training_metadata_table.append(f'{avg_loss:.4f}')

# Build table.
training_metadata_table_frame.grid(row=0, column=0)
for i in range(3):
    training_metadata_table_frame.columnconfigure(i, minsize=75)

for i, label in enumerate(training_metadata_table):
    row = i // 3
    column = i % 3
    tk.Label(training_metadata_table_frame,
             borderwidth=2,
             text=label,
             relief='groove',
             anchor='w').grid(row=row, column=column, sticky='ew')

last_row = len(training_metadata_table) // 3 + 1
tk.Label(training_metadata_table_frame,
         borderwidth=2,
         text=f'Learning rate: {perceptron.learning_rate}',
         relief='groove',
         anchor='w').grid(row=last_row, column=0, columnspan=3, sticky='ew')

# Create image frame.
image_frame = tk.Frame(window)
image_frame.grid(row=2, column=0, sticky='ew')
image_frame_grid_labels = np.empty((EMPTY_GRID.shape[0], EMPTY_GRID.shape[1]), dtype=tk.Label)

sample_input = generate_grid_with_random_pattern()
for i in range(EMPTY_GRID.shape[0]):
    for j in range(EMPTY_GRID.shape[1]):
        color = 'white' if sample_input[i, j] == 1 else 'black'
        image_frame_grid_labels[i, j] = tk.Label(image_frame, bg=color, width=4, height=2)
        image_frame_grid_labels[i, j].grid(row=i, column=j)

def reshuffle_image_frame():
    global sample_input 
    sample_input = generate_grid_with_random_pattern()
    for i in range(EMPTY_GRID.shape[0]):
        for j in range(EMPTY_GRID.shape[1]):
            color = 'white' if sample_input[i, j] == 1 else 'black'
            image_frame_grid_labels[i, j].config(bg=color)

# Create shuffle, predict buttons, and prediction label.
button_frame = tk.Frame(window)
button_frame.grid(row=3, column=0, sticky='ew')

# Create output label.
output_label = tk.Label(button_frame, text='', relief='groove')
output_label.grid(row=1, column=0, columnspan=2, sticky='ew')

def predict_sample():
    global sample_input
    sample_output = perceptron.predict(sample_input.flatten())
    text = 'A' if sample_output == 1 else 'B'
    output_label.config(text=text)
    
predict_sample()

shuffle_button = tk.Button(button_frame, text='Shuffle', width=14, command=reshuffle_image_frame).grid(row=0, column=0, sticky='ew')
predict_button = tk.Button(button_frame, text='Predict', width=14, command=predict_sample).grid(row=0, column=1, sticky='ew')

tk.mainloop()