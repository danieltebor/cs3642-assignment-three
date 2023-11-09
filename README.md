# cs3642-assignment-three
## Information
- Course: CS3642
- Student Name: Daniel Tebor
- Student ID: 000982064
- Assignment #: 3
- Due Date: 11/20
- Signature: Daniel Tebor

## Perceptron Implementation
The core of both problems in this assignment is the perceptron. The perceptron as an input layer of neurons that takes either 0s or 1s as inputs. In addition to the input that can be passed in is an additional bias neuron that is always treated as having a value of one. The following class implements the perceptron:

```python
class BinaryPerceptron:
    def __init__(self, input_size: int, threshold: int, learning_rate: float = 0.1):
        if input_size <= 0:
            raise ValueError('input_size must be greater than 0')
        
        # Create a weight for each input. Add an extra weight for the bias neuron.
        # Initialize all weights to 0.
        self._weights = np.zeros(input_size + 1, dtype=np.float32)
        self._input_size = input_size
        self._threshold = threshold
        self._learning_rate = learning_rate
        
    def predict(self, input: np.ndarray) -> int:
        if len(input) != self._input_size:
            raise ValueError(f'input must be the same size as input_size parameter (expected {self._input_size}, got {len(input)})')
        
        # Append bias neuron.
        input = np.append(input, 1)
        
        # Sum the product of each input by its corresponding weight.
        sum = np.dot(input, self._weights)
        # np.dot(inputs, self._weights) is the same as:
        # sum = 0
        # for i in range(len(inputs)):
        #     sum += inputs[i] * self._weights[i]
        
        # Return 1 if the sum is greater than or equal to the threshold, otherwise return 0.
        return 1 if sum >= self._threshold else 0
    
    def train(self, input: np.ndarray, expected_output: int) -> int:       
        # Calculate the output
        output = self.predict(input)
        
        # Calculate and add the weight delta to each weight.
        delta = self._learning_rate * (expected_output - output)
        # This updates all weights at once instead of looping through each weight.
        self._weights[:-1] += delta * input
        # Add delta to the bias neuron's weight since bias neuron is always 1.
        self._weights[-1] += delta
        
        # Return 1 if the output is correct, otherwise return 0.
        return 1 if output == expected_output else 0
    
    ...
```

The perceptron is initialized with a size for the input layer, a threshold, and a learning rate. The input size is the number of inputs that the perceptron will take. The threshold is the value that the sum of the products between each input and weight must be greater than or equal to in order for the perceptron to return a 1. The weights are initialized to 0 and an addition weight is added to the end of the weights array for the bias neuron. The learning rate scales the weight delta that is added to each weight when the perceptron is trained, effectively speeding up or slowing down training.

The predict method takes an input array of 1s and 0s with the same size as the designated input size. The input array is appended with a 1 to represent the bias neuron. To determine the output, the input value of each neuron is multiplied by its corresponding weight and the products are summed. If the sum is greater than or equal to the threshold, the perceptron returns a 1, otherwise it returns a 0.

The train method takes an input and an expected output. The predict method is called to predict the output. Next the weight delta is calculated by multiplying the learning rate by the difference between the expected output and the actual output. The delta for each weight is calculated by subtracting the expected output by the actual output, multiplying it by the learning rate and the input value. If the output and expected output match, the weights will not be adjusted as the delta is multiplied by zero. Finally 1 is returned if the model predicted correctly, otherwise 0 is returned.

## Problem 1: Bright or Dark Image Discernment
The goal for problem one is to train the perceptron to discern between bright and dark images. The images are a 2x2 matrix of either white (represented as 1) or black (represented as 0) pixels. If the images has 2 or more white tiles, the tile is considered bright, otherwise dark.

Training the perceptron requires a training sample. In this case it is all possible 2x2 images with white or black pixels. The following code generates the training sample:

```python
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
```

In order to determine if the perceptron is trained correctly, each input image is assigned a label that corresponds to whether the image is bright or dark. Bright is represented by 1 and dark is represented by 0. Ironically, this approach uses a typical boolean function to determine if the image is bright or dark.

Next, the perceptron is initialized. Each image has 4 pixels, so the input size to the perceptron is 4. The threshold for whether the image is bright or dark is 2 so the threshold is set to 2. A learning rate of 0.1 was determined to converge the algorithm very quickly to 100% accuracy.

```python
perceptron = BinaryPerceptron(input_size=4, threshold=2, learning_rate=0.1)
```

Finally, the perceptron is trained using the training sample. Each epoch, the perceptron is trained with each image in the training sample. The perceptron is trained until the calculated average loss reaches 0, meaning the perceptron was 100% accurate. The loss is calculated by summing up the loss for each epoch and then dividing by the number of inputs. If the model predicts incorrectly, 1 is added to the loss, otherwise nothing is added. The following code trains the perceptron:

```python
for i in range(1, 1000):
    avg_loss = 0
    
    # Train the perceptron.
    for input, label in data:
        avg_loss += 1 - perceptron.train(input=input, expected_output=label)
        
    avg_loss /= len(images)
    
    # GUI stuff
    # ...
    # End GUI stuff
    
    if avg_loss == 0:
        break
```

### Visualization
Tkinter was used to make a basic GUI to visualize the perceptron. The GUI displays training metadata including the learning rate and the model weights and the loss for each epoch. The GUI also allows the user to shuffle an image and see the trained perceptron's prediction.

![](/assets/problem-one-gui-example.png)

The following code handles the GUI:

```python
# Create window.                
window = tk.Tk()
window.title('Question 1')
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
```

## Problem 2: Pattern Recognition
The goal for problem two was to try and reach the limit of what a perceptron could predict with 100% accuracy. For this problem, the perceptron was trained to recognize capital A or capital B letters in an 8x8 image of black or white pixels. The letters had a small and large pattern as well, the representation of which can be seen in the following code:

```python
# Letter A patterns.
A_PATTERNS = [
    [
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0]
    ],
    [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0]
    ]
]

# Letter B patterns.
B_PATTERNS = [
    [
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1]
    ],
    [
        [0, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1]
    ]
]
```

Each image had one of the patterns placed in the image in a random location as long as the pattern fit in the bounds of the image. To generate the training sample for each epoch, each pattern was randomly placed in an image, resulting in 4 images per epoch. The following code handles the creation of training samples and standalone images:

```python
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
```

The randomly_place_pattern_on_grid function takes a grid and a pattern and places the pattern in a random location on the grid. This is used by both the generate_training_data and generate_grid_with_random_pattern functions. The former generates a training sample of 4 images and labels, while the latter generates a single image with a random pattern. The A patterns were labeled as 1 and the B patterns were labeled as 0.

The perceptron was initialized with an input size of 64 (8x8 image). A threshold of 16 and a learning rate of 0.01 which were determined to allow the perceptron to converge to 100% accuracy quickly.

```python
perceptron = BinaryPerceptron(input_size=EMPTY_GRID.shape[0] * EMPTY_GRID.shape[1], threshold=16, learning_rate=0.01)
```

Training the perceptron was similar to problem one. The perceptron was trained with each image in the training sample until the average loss was 0. The biggest difference was the use of randomly generated images instead of a static training sample. Over many epochs this allows the perceptron to learn the patterns for each letter without needing a static dataset of every pattern in every possible location. In addition, if the loss reaches 0, the perceptron was tested on 99996 randomly generated images to ensure that the perceptron was 100% accurate. If any of the images was found to be incorrect, training would continue. The following code trains the perceptron:

```python
for i in range(1, MAX_EPOCHS):
    inputs, labels = generate_training_data()
    avg_loss = 0

    # Train the perceptron.
    for input, label in zip(inputs, labels):
        avg_loss += 1 - perceptron.train(input=input, expected_output=label)
        
    avg_loss /= len(inputs)

    # GUI stuff
    # ...
    # End GUI stuff

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
```

### Visualization
Like in Problem 1, Tkinter was used to make a basic GUI to visualize the perceptron. The GUI displays training metadata including the learning rate and the model weights and the loss for each epoch. The GUI also allows the user to generate a random image with a randomly selected pattern and location and see the trained perceptron's prediction.

![](/assets/problem-two-gui-example.png)

The following code handles the GUI:

```python
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
```

## Building and Running
Python 3.10.12 was used to build the project. It is likely that other versions also work.

### Prerequisites
To build and run the project, the following prerequisites are required:

- Python 3.10.12+ (https://www.python.org/downloads/)
- Tkinter (https://docs.python.org/3/library/tkinter.html)
- Python Pip (https://pip.pypa.io/en/stable/installation/)

To install the required pip packages, run the following command in project root:

```bash
pip install -r requirements.txt
```

### Running
problem_one.py or problem_two.py can be run directly from the command line in project root or with a properly configured IDE. The following commands can be used to run the programs.