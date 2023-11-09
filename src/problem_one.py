import numpy as np
import tkinter as tk

from binary_perceptron import BinaryPerceptron


# Create all input combinations for 2x2 image.
images = []
labels = []

for i in range (0, 2):
    for j in range (0, 2):
        for k in range (0, 2):
            for l in range (0, 2):
                images.append([i, j, k, l])
                labels.append(1 if i + j + k + l >= 2 else 0)

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int8)
data = list(zip(images, labels))

# Create window.                
window = tk.Tk()
window.title('Problem 1')
window.geometry('500x600')

training_metadata_table_frame = tk.Frame(window)
training_metadata_table_frame.grid(row=0, column=0)
training_metadata_table = []

# Train the perceptron.
perceptron = BinaryPerceptron(input_size=4, threshold=2, learning_rate=0.1)

training_metadata_table.append('Epoch')
training_metadata_table.append('Weights')
training_metadata_table.append('Avg Loss')
training_metadata_table.append('-')
training_metadata_table.append('[0.00, 0.00, 0.00, 0.00, 0.00]')
avg_loss = 0
for input, _ in data:
    avg_loss += 1 - perceptron.predict(input)
avg_loss /= len(images)
training_metadata_table.append(avg_loss)

# Iterate over a number of epochs.
for i in range(1, 1000):
    avg_loss = 0
    
    # Train the perceptron.
    for input, label in data:
        avg_loss += 1 - perceptron.train(input=input, expected_output=label)
        
    avg_loss /= len(images)
    
    formatted_weights = '['
    for weight in perceptron.weights:
        formatted_weights += f'{weight:.2f}, '
    formatted_weights = formatted_weights[:-2] + ']'
    
    training_metadata_table.append(i)
    training_metadata_table.append(formatted_weights)
    training_metadata_table.append(f'{avg_loss:.4f}')
    
    if avg_loss == 0:
        break
    
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
image_frame_grid_labels = np.empty((2, 2), dtype=tk.Label)

sample_input = np.random.randint(0, 2, size=(2, 2))
for i in range(2):
    for j in range(2):
        color = 'white' if sample_input[i, j] == 1 else 'black'
        image_frame_grid_labels[i, j] = tk.Label(image_frame, bg=color, width=16, height=8)
        image_frame_grid_labels[i, j].grid(row=i, column=j)

def reshuffle_image_frame():
    global sample_input 
    sample_input = np.random.randint(0, 2, size=(2, 2))
    for i in range(2):
        for j in range(2):
            color = 'white' if sample_input[i, j] == 1 else 'black'
            image_frame_grid_labels[i, j].config(bg=color)

# Create shuffle, predict buttons, and prediction label.
button_frame = tk.Frame(window)
button_frame.grid(row=3, column=0, sticky='ew')

# Create output label.
output_label = tk.Label(button_frame, text='', relief='groove')
output_label.grid(row=1, column=0, columnspan=2, sticky='ew')

def predict_sample():
    sample_output = perceptron.predict(sample_input.flatten())
    text = 'Bright' if sample_output == 1 else 'Dark'
    output_label.config(text=text)
    
predict_sample()

shuffle_button = tk.Button(button_frame, text='Shuffle', width=13, command=reshuffle_image_frame).grid(row=0, column=0, sticky='ew')
predict_button = tk.Button(button_frame, text='Predict', width=13, command=predict_sample).grid(row=0, column=1, sticky='ew')

tk.mainloop()