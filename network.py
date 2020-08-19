import random
import numpy as np

sigmoid = lambda x : 1.0 / (1.0 + np.exp(-x))
sigmoid_prime = lambda x : np.exp(-x) / (1.0 + np.exp(-x)) / (1.0 + np.exp(-x))

class Network(object):
  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

  def feedforward(self, input):
    for w, b in zip(self.weights, self.biases):
      input = sigmoid(np.dot(w, input) + b)
    return input
  
  def evaluate(self, test_data):
    test_results = [[np.argmax(self.feedforward(x)), y] for x, y in test_data]
    return sum(int(x == y) for x, y in test_results)

  def cost_derivative(self, output, y):
    return output - y

  def backprop(self, x, y):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    activation, activations, zs = x, [x], []
    for w, b in zip(self.weights, self.biases):
      z = np.dot(w, activation) + b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    for l in range(2, self.num_layers):
      z = zs[-l]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
    return [nabla_b, nabla_w]

  def update_network(self, mini_batch, step):
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    for x, y in mini_batch:
      cur_nabla_w, cur_nabla_b = self.backprop(x, y)
      nabla_w = [now + add for now, add in zip(nabla_w, cur_nabla_w)]
      nabla_b = [now + add for now, add in zip(nabla_b, cur_nabla_b)]
    self.weights = [w - step * change / len(mini_batch) for w, change in zip(self.weights, nabla_w)]
    self.weights = [b - step * change / len(mini_batch) for b, change in zip(self.biases, nabla_b)]

  def descend(self, training_data, epochs, mini_batch_size, step, test_data = None):
    if test_data: num_test = len(test_data)
    n = len(training_data)
    for it in range(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[i : i + mini_batch_size] for i in range(0, n, mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_network(mini_batch, step)
      if test_data:
        print("Epoch {}: {} / {}".format(it, self.evaluate(test_data), num_test))
      else:
        print("Epoch {} completed".format(it))    


