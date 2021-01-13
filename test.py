import numpy as np
from NeuralNetwork import NeuralNetwork

NN1 = NeuralNetwork("model1")

NN1.add_layer(2)
NN1.add_layer(3)
NN1.add_layer(2)

expected_outputs = []
inputs = []

for i in range(50000):
    _input = np.random.randint(0, 2, size=(2,1))
    inputs.append(_input)
    if _input[0][0] == 1 and _input[1][0] == 1:
        expected_outputs.append( np.array([[1], [0]]) )
    else:
        expected_outputs.append( np.array([[0], [1]]) )

NN1.train(expected_outputs, inputs, NeuralNetwork.cross_entropy_loss)

NeuralNetwork.save_state(NN1)

NN2 = NeuralNetwork("model2")

NN2.add_layer(2)
NN2.add_layer(3)
NN2.add_layer(2)

NN2.train(expected_outputs, inputs, NeuralNetwork.cross_entropy_loss)

NeuralNetwork.save_state(NN2)