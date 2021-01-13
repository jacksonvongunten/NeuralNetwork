from NeuralNetwork import NeuralNetwork
import numpy as np

NN1 = NeuralNetwork.load_state("network_model1.pkl")
NN2 = NeuralNetwork.load_state("network_model2.pkl")
inputs = np.array([[1], [0]])

res1 = NN1.forward_propogate(inputs)
res2 = NN2.forward_propogate(inputs)