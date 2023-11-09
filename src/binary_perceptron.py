import numpy as np


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
        # Add delta to the bias neuron since bias neuron is always 1.
        self._weights[-1] += delta
        
        # Return 1 if the output is correct, otherwise return 0.
        return 1 if output == expected_output else 0
    
    @property
    def input_size(self) -> int:
        return self._input_size
    
    @property
    def weights(self) -> np.ndarray:
        return np.copy(self._weights)
    
    @property
    def threshold(self) -> int:
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: int):
        self._threshold = value
        
    @property
    def learning_rate(self) -> float:
        return self._learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value: float):
        self._learning_rate = value