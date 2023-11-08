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
                images.append(np.array([i, j, k, l], dtype=np.float32))
                labels.append(i + j + k + l >= 2)
                
data = list(zip(images, labels))
                
# Create window.                
window = tk.Tk()
window.title('Question 1')
window.geometry('500x500')

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
for input, label in zip(images, labels):
    avg_loss += 1 - perceptron.train(input, label)
avg_loss /= len(images)
training_metadata_table.append(avg_loss)

for i in range(1, 1000):
    avg_loss = 0
    
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

sample_input = np.zeros((2, 2), dtype=np.int32)

def shuffle_image_frame():
    sample_input = np.random.randint(0, 2, size=(2, 2))
    for i in range(2):
        for j in range(2):
            color = 'white' if sample_input[i, j] == 1 else 'black'
            tk.Label(image_frame, bg=color, width=10, height=5).grid(row=i, column=j)

shuffle_image_frame()

output_label = tk.Label(window, text='', width=10, height=5)
output_label.grid(row=4, column=0)

def predict_sample():
    sample_output = perceptron.predict(sample_input.flatten())
    text = 'Bright' if sample_output == 1 else 'Dark'
    print(text)
    output_label.config(text=text)
    
predict_sample()

# Create shuffle and predict buttons.
button_frame = tk.Frame(window)
button_frame.grid(row=3, column=0, sticky='ew')

shuffle_button = tk.Button(button_frame, text='Shuffle', width=6, command=shuffle_image_frame).grid(row=0, column=0, sticky='ew')
predict_button = tk.Button(button_frame, text='Predict', width=6, command=predict_sample).grid(row=0, column=1, sticky='ew')

tk.mainloop()
    