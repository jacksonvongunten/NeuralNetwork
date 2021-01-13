import numpy as np
import pickle
from timeit import default_timer as timer

class NeuralNetwork:

    @staticmethod
    def sigmoid(*args):
        #args[0] is the input
        #args[1] False: calculate activation. True: calculate derivative of activation
        if not args[1]:
            return 1 / (1 + np.exp(-args[0]))
        else:
            return NeuralNetwork.sigmoid(args[0], False)*(1-NeuralNetwork.sigmoid(args[0], False))
        
    @staticmethod
    def squared_error_loss(*args):
        #args[0] is the expected value
        #args[1] is the value received
        #args[2] False: calculate error. True: calculate derivative of error
        t = args[0]
        y = args[1]
        if not args[2]:
            return 0.5*(t-y)**2
        elif args[2]:
            return y-t

    @staticmethod 
    def cross_entropy_loss(*args):
        #args[0] is the expected value
        #args[1] is the value received
        #args[2] False: calculate error. True: calculate derivative of error
        t = args[0]
        y = args[1]
        if not args[2]:
            return t*np.ln(y) + (1-t)*np.ln(1-y)
        elif args[2]:
            return y-t

    @staticmethod
    def save_state(obj):
        with open("network_" + obj.name + ".pkl", "wb") as f:
            pickle.dump(obj, f)
    
    @staticmethod
    def load_state(filename):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj

    def __init__(self, name):
        self.name = name
        self.weighted_inputs = []
        self.activations = []
        
        self.num_layers = 0
        self.num_nodes_per_layer = []

        self.weights = []
        self.biases = []

        self.learning_rate = 0.1
        self.error = NeuralNetwork.squared_error_loss
        self.activation = NeuralNetwork.sigmoid
    
    def __repr__(self):
        string = f"Network \"{self.name}\" consists of {self.num_layers} layers\n"
        string += f"Inputs: {len(self.activations[0])}, Outputs: {len(self.activations[self.num_layers-1])}\n\n"
        for i in range(self.num_layers-1):
            string += f"Weight Matrix {i} of Shape {self.weights[i].shape}\n"
            string += f"{self.weights[i]}\n\n"
            string += f"Bias Matrix {i} of Shape {self.biases[i].shape}\n"
            string += f"{self.biases[i]}\n\n"
        return string.rstrip()

    def add_layer(self, num_nodes):
        self.num_layers += 1
        self.num_nodes_per_layer.append(num_nodes)
        self.activations.append( np.zeros((num_nodes, 1)) )
        self.weighted_inputs.append( np.zeros((num_nodes, 1)) )

        if self.num_layers > 1:
            self.weights.append( np.random.rand( self.num_nodes_per_layer[self.num_layers-1], self.num_nodes_per_layer[self.num_layers-2]) )
            self.biases.append( np.random.rand( self.num_nodes_per_layer[self.num_layers-1], 1) )

    def forward_propogate(self, inputs):
        self.weighted_inputs[0] = inputs
        self.activations[0] = self.weighted_inputs[0]
        for i in range(1, self.num_layers):
            self.weighted_inputs[i] = np.dot(self.weights[i-1], self.weighted_inputs[i-1]) + self.biases[i-1]
            self.activations[i] = self.activation( self.weighted_inputs[i], False )
        return self.activations[self.num_layers-1]
    
    def error(self, expected_output, inputs):
        output = self.forward_propogate(inputs)
        return self.error(expected_output, inputs, False)
    
    def d_error(self, expected_output, inputs):
        output = self.forward_propogate(inputs)
        return self.error(expected_output, output, True)

    def backpropogate(self, expected_output, inputs, loss_function=None):
        if loss_function is None:
            loss_function = NeuralNetwork.squared_error_loss
        else:
            self.error = loss_function
        
        if self.error == NeuralNetwork.squared_error_loss:
            current_layer = self.num_layers-1
            delta_L = self.d_error(expected_output, inputs) * self.activation(self.weighted_inputs[current_layer], True)
            self.weights[current_layer-1] -= self.learning_rate*np.dot( delta_L, self.activations[current_layer-1].T )
            self.biases[current_layer-1] -= self.learning_rate*delta_L

            current_layer -= 1
            while current_layer >= 1:
                delta_L = np.dot( self.weights[current_layer].T, delta_L ) * self.activation(self.weighted_inputs[current_layer], True)
                self.weights[current_layer-1] -= self.learning_rate*np.dot( delta_L, self.activations[current_layer-1].T )
                self.biases[current_layer-1] -= self.learning_rate*delta_L
                current_layer -= 1
        
        elif self.error == NeuralNetwork.cross_entropy_loss:
            current_layer = self.num_layers-1
            delta_L = self.d_error(expected_output, inputs)
            self.weights[current_layer-1] -= self.learning_rate*np.dot( delta_L, self.activations[current_layer-1].T )
            self.biases[current_layer-1] -= self.learning_rate*delta_L

            current_layer -= 1
            while current_layer >= 1:
                delta_L = np.dot( self.weights[current_layer].T, delta_L )
                self.weights[current_layer-1] -= self.learning_rate*np.dot( delta_L, self.activations[current_layer-1].T )
                self.biases[current_layer-1] -= self.learning_rate*delta_L
                current_layer -= 1

    def train(self, expected_outputs, inputs, loss_function=None):
        start = timer()
        for (o, i) in zip(expected_outputs, inputs):
            self.backpropogate(o, i, loss_function)
        end = timer()
        print(f'Trained on {len(expected_outputs)} examples in {end-start} seconds')