from random import seed
import random
import math
from math import exp
from PIL import Image
import numpy as np

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random.uniform(-(math.sqrt(6.0) / math.sqrt(n_inputs + (n_hidden[1] if len(n_hidden) > 1 else n_outputs))), (math.sqrt(6.0) / math.sqrt(n_inputs + (n_hidden[1] if len(n_hidden) > 1 else n_outputs)))) for i in range(n_inputs + 1)]} for i in range(n_hidden[0])]
    network.append(hidden_layer)
    for j in range(1, len(n_hidden)):
        hidden_layer = [{'weights': [random.uniform(-(math.sqrt(6.0) / math.sqrt(n_hidden[j-1] + (n_hidden[j+1] if len(n_hidden) > j else n_outputs))), (math.sqrt(6.0) / math.sqrt(n_hidden[j-1] + (n_hidden[j+1] if len(n_hidden) > j else n_outputs)))) for i in range(n_hidden[j - 1] + 1)]} for i in range(n_hidden[j])]
        network.append(hidden_layer)
    output_layer = [{'weights': [random.uniform(-(math.sqrt(6.0) / math.sqrt(n_hidden[len(n_hidden)-1])), (math.sqrt(6.0) / math.sqrt(n_hidden[len(n_hidden)-1]))) for i in range(n_hidden[-1] + 1)]} for k in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, outs, l_rate, n_epoch, n_outputs):
    correct = 0
    for epoch in range(n_epoch):
        sum_error = 0
        for j in range(len(train)):
            outputs = forward_propagate(network, train[j])
            expected = [0 for i in range(n_outputs)]
            expected[outs[j]] = 1
            if (np.argmax(outputs) == outs[j]):
                correct += 1
            sum_error += sum([(expected[i] - outputs[i]) for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, train[j], l_rate)
        accuracy = correct / ((epoch + 1) * len(train))
        sum_error /= 2
        print('>epoch=%d, error=%.3f, accuracy=%.4f' % (epoch, sum_error, accuracy))


def read_img(name):
    img = Image.open(name)
    img = img.resize((7, 7))
    img = img.convert('L')
    img = np.array(img)
    # img = img.reshape(1, 7, 7, 1)
    img = img / 255.0
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] > 0.6:
                img[i][j] = 1
            elif img[i][j] < 0.4:
                img[i][j] = 0
    return img.flatten()


# Make a prediction with a network
def predict(network, dataset):
    output = []
    for row in dataset:
        y = forward_propagate(network, row)
        # print(str(row)+" "+str(y[0]))
        output.append(y)
    return output

# def write_data(name,train,out):
#     with open(name,'w',encoding='UTF8') as f:
#         writer=csv.writer(f)
#         for i in range(len(train)):
#             writer.writerow(np.append(train[i],out[i]))


train_images = []
train_images.append(read_img("circle.png"))
train_images.append(read_img("square.png"))
train_images.append(read_img("triangle.png"))
train_images = np.array(train_images)

train_figure = [0, 1, 2]

class_names = ['Circle', 'Square', 'Triangle']

# write_data('dataset.csv',train_images,train_figure)

seed(1)
n_hidden = [12]
interation = 1000
n_outputs = len(set(train_figure))
n_inputs = len(train_images[0])

network = initialize_network(n_inputs, n_hidden, n_outputs)

train_network(network, train_images, train_figure, 0.2, interation, n_outputs)

print("Predict data")
predictions = predict(network, np.array(train_images))

for i in range(len(predictions)):
    print(predictions[i])
    print(class_names[np.argmax(predictions[i])])

print("Test data")
# test
test_images = []

test_images.append(read_img("test_square.png"))
test_images.append(read_img("test_triangle.png"))
test_images.append(read_img("test_circle.png"))
test_images.append(read_img("test_triangle_reverse.png"))

test_images = np.array(test_images)

predictions = predict(network, test_images)

for i in range(len(predictions)):
    print(predictions[i])
    print(class_names[np.argmax(predictions[i])])
